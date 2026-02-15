import base64
import io
import numpy as np
import librosa
import binascii
import os


# -----------------------------
# Audio trimming and padding
# -----------------------------
def trim_audio(waveform, sr, max_duration=12):
    max_samples = int(max_duration * sr)

    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]
    else:
        pad_amount = max_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_amount))

    return waveform



# -----------------------------
# Input type detection
# -----------------------------
def detect_audio_input_type(input_value: str) -> str:
    """
    Returns one of:
    - 'wav_path'
    - 'mp3_path'
    - 'base64_wav'
    - 'base64_mp3'
    Raises ValueError if unknown.
    """

    # File path check
    if os.path.isfile(input_value):
        ext = os.path.splitext(input_value)[1].lower()

        if ext == ".wav":
            return "wav_path"
        elif ext == ".mp3":
            return "mp3_path"
        else:
            raise ValueError("Unsupported file format")

    # Try Base64 decode
    try:
        audio_bytes = base64.b64decode(input_value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Input is neither a valid file path nor Base64")

    # Detect WAV
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]:
        return "base64_wav"

    # Detect MP3
    if (
        audio_bytes.startswith(b"ID3")
        or audio_bytes[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]
    ):
        return "base64_mp3"

    raise ValueError("Unsupported audio format")


# -----------------------------
# Unified preprocessing
# -----------------------------
def preprocess_audio(input_value: str, target_sr: int = 16000):
    input_type = detect_audio_input_type(input_value)

    # Base64 case
    if input_type in ["base64_wav", "base64_mp3"]:
        audio_bytes = base64.b64decode(input_value)
        buffer = io.BytesIO(audio_bytes)
        waveform, _ = librosa.load(buffer, sr=target_sr, mono=True)
    else:
        waveform, _ = librosa.load(input_value, sr=target_sr, mono=True)

    # remove leading/trailing silence
    waveform, _ = librosa.effects.trim(waveform, top_db=30)

    # fixed-length trim/pad
    waveform = trim_audio(waveform, sr=target_sr)

    # normalize amplitude
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-9)

    return waveform.astype(np.float32), target_sr

