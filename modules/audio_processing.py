
import librosa
import numpy as np


def analyze_audio(audio_file):
    """Extract basic acoustic measurements from an audio file."""

    y, sr = librosa.load(audio_file, sr=16000, mono=True)

    if len(y) == 0:
        raise ValueError("The uploaded audio file is empty.")

    duration = len(y) / sr

    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=1.0)

    intervals = librosa.effects.split(
        y,
        top_db=30,
        frame_length=512,
        hop_length=128
    )

    speech_seconds = sum(
        end - start for start, end in intervals
    ) / sr

    pauses = []
    previous_end = 0

    for start, end in intervals:
        pause_duration = (start - previous_end) / sr

        if pause_duration >= 0.25:
            pauses.append({
                "start": round(previous_end / sr, 2),
                "end": round(start / sr, 2),
                "duration": round(pause_duration, 2)
            })

        previous_end = end

    trailing_pause = (len(y) - previous_end) / sr

    if trailing_pause >= 0.25:
        pauses.append({
            "start": round(previous_end / sr, 2),
            "end": round(duration, 2),
            "duration": round(trailing_pause, 2)
        })

    f0 = librosa.yin(
        y,
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=2048
    )

    voiced_pitch = f0[np.isfinite(f0)]

    return {
        "duration": round(duration, 2),
        "estimated_speech_seconds": round(speech_seconds, 2),
        "pause_count": len(pauses),
        "pauses": pauses,
        "average_energy_db": round(float(np.mean(rms_db)), 2),
        "pitch_median_hz": (
            round(float(np.median(voiced_pitch)), 2)
            if len(voiced_pitch) else None
        ),
        "pitch_variation_hz": (
            round(float(np.std(voiced_pitch)), 2)
            if len(voiced_pitch) else None
        )
    }
