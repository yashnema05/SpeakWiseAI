
import re
from functools import lru_cache

import numpy as np
import soundfile as sf
import torch
import torchaudio

from torchaudio.pipelines import MMS_FA


@lru_cache(maxsize=1)
def _load_alignment_model():
    """Load the multilingual alignment model once."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MMS_FA.get_model().to(device)
    model.eval()

    tokenizer = MMS_FA.get_tokenizer()
    aligner = MMS_FA.get_aligner()

    return model, tokenizer, aligner, device


def forced_align(audio_file, transcript):
    """
    Align a supplied transcript to an audio recording.

    Hindi should currently be entered in Romanized form.
    The transcript must match the spoken words.
    """

    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Please provide the complete speech transcript.")

    # Keep supported English/Romanized words.
    words = re.findall(r"[a-zA-Z']+", transcript.lower())

    if not words:
        raise ValueError(
            "No supported words found. Enter English or Romanized Hindi, "
            "for example: namaste aap kaise hain."
        )

    try:
        # Read audio without torchaudio.load / TorchCodec.
        audio, sample_rate = sf.read(
            audio_file,
            dtype="float32",
            always_2d=True
        )
    except Exception as exc:
        raise ValueError(
            f"Could not read audio file: {exc}"
        ) from exc

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    # Convert stereo or multichannel audio to mono.
    audio = audio.mean(axis=1)

    # Convert NumPy audio to a PyTorch tensor.
    waveform = torch.from_numpy(
        np.ascontiguousarray(audio)
    ).unsqueeze(0)

    # Resample to the model's required sample rate.
    if sample_rate != MMS_FA.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=MMS_FA.sample_rate
        )

    duration = waveform.shape[1] / MMS_FA.sample_rate

    model, tokenizer, aligner, device = _load_alignment_model()

    try:
        tokens = tokenizer(words)

        if not tokens or any(len(token) == 0 for token in tokens):
            raise ValueError("Could not tokenize one or more words.")

        with torch.inference_mode():
            emissions, _ = model(waveform.to(device))
            word_spans = aligner(emissions[0], tokens)

    except Exception as exc:
        raise RuntimeError(
            "The alignment model could not process this transcript. "
            "Check the transcript and model compatibility. "
            f"Details: {exc}"
        ) from exc

    # Convert model-frame indices to seconds.
    seconds_per_frame = (
        duration / emissions.shape[1]
        if emissions.shape[1] > 0 else 0
    )

    results = []

    for word, spans in zip(words, word_spans):
        if not spans:
            results.append({
                "word": word,
                "start": None,
                "end": None,
                "score": None
            })
            continue

        start_time = spans[0].start * seconds_per_frame
        end_time = spans[-1].end * seconds_per_frame

        scores = [
            float(span.score)
            for span in spans
            if hasattr(span, "score")
        ]

        results.append({
            "word": word,
            "start": round(max(0.0, start_time), 3),
            "end": round(min(duration, end_time), 3),
            "score": (
                round(float(np.mean(scores)), 4)
                if scores else None
            )
        })

    return results
