import streamlit as st
import torch
import librosa
from transformers import pipeline


@st.cache_resource(show_spinner=False)
def load_transcription_model():
    """Load multilingual Whisper once and reuse it."""
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-base",
        device=device
    )


def transcribe_audio(audio_path):
    """
    Automatically transcribe uploaded speech.
    Supports multilingual speech, including Hindi and English.
    """
    if not audio_path:
        raise ValueError("Please upload an audio file.")

    try:
        audio, sample_rate = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        if len(audio) == 0:
            raise ValueError("The uploaded audio is empty.")

        recognizer = load_transcription_model()

        result = recognizer(
            {
                "raw": audio,
                "sampling_rate": sample_rate
            },
            chunk_length_s=30
        )

        transcript = result.get("text", "").strip()

        if not transcript:
            raise ValueError(
                "No speech could be transcribed. Try a clearer recording."
            )

        return transcript

    except Exception as exc:
        raise RuntimeError(
            f"Automatic transcription failed: {exc}"
        ) from exc
