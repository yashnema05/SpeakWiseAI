
from pathlib import Path
import json

import librosa
import numpy as np
import soundfile as sf
import pyworld


ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / "reference"
FLAWED_DIR = ROOT / "flawed"

REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
FLAWED_DIR.mkdir(parents=True, exist_ok=True)

reference_path = REFERENCE_DIR / "reference.wav"

if not reference_path.exists():
    raise FileNotFoundError("Missing dataset/reference/reference.wav")

audio, sr = librosa.load(str(reference_path), sr=16000, mono=True)

if len(audio) == 0:
    raise ValueError("Reference audio is empty.")

audio = audio.astype(np.float64)
test_cases = []


# 1. EXTRA SILENCE
insert_at = len(audio) // 2
silence_duration = 1.5
silence_samples = int(silence_duration * sr)

extra_silence_audio = np.concatenate([
    audio[:insert_at],
    np.zeros(silence_samples),
    audio[insert_at:]
])

silence_start = insert_at / sr
silence_end = silence_start + silence_duration

sf.write(
    str(FLAWED_DIR / "extra_silence.wav"),
    extra_silence_audio,
    sr
)

test_cases.append({
    "audio_file": "dataset/flawed/extra_silence.wav",
    "reference_file": "dataset/reference/reference.wav",
    "flaws": [{
        "flaw": "Extra Silence",
        "start": round(silence_start, 3),
        "end": round(silence_end, 3),
        "severity": 3
    }]
})


# 2. MONOTONE SPEECH - WORLD F0 FLATTENING
frame_period = 5.0

_f0, time_axis = pyworld.harvest(
    audio,
    sr,
    frame_period=frame_period
)

f0 = pyworld.stonemask(
    audio,
    _f0,
    time_axis,
    sr
)

spectrogram = pyworld.cheaptrick(
    audio,
    f0,
    time_axis,
    sr
)

aperiodicity = pyworld.d4c(
    audio,
    f0,
    time_axis,
    sr
)

voiced_f0 = f0[f0 > 0]

if len(voiced_f0) == 0:
    raise ValueError("No voiced pitch frames found in reference audio.")

# Replace voiced pitch variation with one constant median F0.
constant_f0 = float(np.median(voiced_f0))
flat_f0 = np.where(f0 > 0, constant_f0, 0.0)

monotone_audio = pyworld.synthesize(
    flat_f0,
    spectrogram,
    aperiodicity,
    sr,
    frame_period=frame_period
)

monotone_audio = monotone_audio.astype(np.float32)

sf.write(
    str(FLAWED_DIR / "monotone_speech.wav"),
    monotone_audio,
    sr
)

test_cases.append({
    "audio_file": "dataset/flawed/monotone_speech.wav",
    "reference_file": "dataset/reference/reference.wav",
    "flaws": [{
        "flaw": "Monotone Speech",
        "start": 0.0,
        "end": round(len(monotone_audio) / sr, 3),
        "severity": 3
    }]
})


# 3. ABNORMAL PACING - SLOWER SPEECH
stretched_audio = librosa.effects.time_stretch(
    audio.astype(np.float32),
    rate=0.75
)

sf.write(
    str(FLAWED_DIR / "abnormal_pacing.wav"),
    stretched_audio,
    sr
)

test_cases.append({
    "audio_file": "dataset/flawed/abnormal_pacing.wav",
    "reference_file": "dataset/reference/reference.wav",
    "flaws": [{
        "flaw": "Abnormal Pacing",
        "start": 0.0,
        "end": round(len(stretched_audio) / sr, 3),
        "severity": 3
    }]
})


# SAVE GROUND TRUTH
ground_truth = {"test_cases": test_cases}

with open(ROOT / "ground_truth.json", "w", encoding="utf-8") as file:
    json.dump(ground_truth, file, indent=4)


print("\nDataset generation completed successfully!")
print(f"Constant F0 used: {constant_f0:.2f} Hz")
print(f"Reference duration: {len(audio) / sr:.2f} sec")
print(f"Monotone duration: {len(monotone_audio) / sr:.2f} sec")
print(f"Extra silence: {silence_duration:.2f} sec")
print(f"Abnormal pacing rate: 0.75")
print("\nGround truth saved.")
