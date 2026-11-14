
import librosa
import numpy as np


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=16000, mono=True)

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    duration = len(y) / sr
    hop_length = 160
    frame_length = 1024

    # Pitch
    f0 = librosa.yin(
        y,
        fmin=65,
        fmax=400,
        sr=sr,
        frame_length=2048,
        hop_length=hop_length
    )

    # Short-time energy and MFCCs
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        hop_length=hop_length
    )

    # Approximate speech regions. These are energy-based estimates.
    intervals = librosa.effects.split(
        y,
        top_db=30,
        frame_length=512,
        hop_length=128
    )

    speech_seconds = sum(
        end - start for start, end in intervals
    ) / sr

    # Convert gaps between speech regions into pause intervals.
    # Include leading and trailing silence where present.
    pause_intervals = []
    previous_end = 0

    for start, end in intervals:
        gap = (start - previous_end) / sr

        if gap >= 0.25:
            pause_intervals.append({
                "start": round(previous_end / sr, 2),
                "end": round(start / sr, 2),
                "duration": round(gap, 2)
            })

        previous_end = max(previous_end, end)

    trailing_gap = (len(y) - previous_end) / sr

    if trailing_gap >= 0.25:
        pause_intervals.append({
            "start": round(previous_end / sr, 2),
            "end": round(duration, 2),
            "duration": round(trailing_gap, 2)
        })

    # One-second speech activity windows
    window_samples = sr
    pace_times = []
    speech_activity = []

    for start in range(0, len(y), window_samples):
        end = min(start + window_samples, len(y))
        window = y[start:end]

        if len(window) < int(0.1 * sr):
            continue

        active_intervals = librosa.effects.split(
            window,
            top_db=30,
            frame_length=512,
            hop_length=128
        )

        active_samples = sum(
            b - a for a, b in active_intervals
        )

        pace_times.append((start + len(window) / 2) / sr)
        speech_activity.append(
            float(active_samples / max(len(window), 1))
        )

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    pitch_times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length
    )

    return {
        "times": times,
        "pitch_times": pitch_times,
        "pitch": f0,
        "energy": rms,
        "mfcc": mfcc,
        "duration": round(duration, 2),
        "estimated_speech_seconds": round(speech_seconds, 2),
        "speech_ratio": round(speech_seconds / duration, 3),
        "pace_times": pace_times,
        "speech_activity": speech_activity,
        "pause_intervals": pause_intervals,
        "pause_count": len(pause_intervals)
    }
