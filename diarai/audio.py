"""Audio decoding: files (any format PyAV reads) and streamed chunks (WAV or raw PCM)."""

import io
import wave
from pathlib import Path
from typing import Union

import numpy as np

from diarai.config import SAMPLE_RATE


def load_audio(audio: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """Decode any audio/video file (path or bytes) to 16 kHz mono float32."""
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    from faster_whisper import decode_audio

    src = io.BytesIO(audio) if isinstance(audio, (bytes, bytearray)) else str(audio)
    return decode_audio(src, sampling_rate=SAMPLE_RATE)


def resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE or len(x) == 0:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, SAMPLE_RATE)
    return resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)


def decode_chunk(data: bytes, sample_rate: int = SAMPLE_RATE, codec: str = "pcm_s16le") -> np.ndarray:
    """One streamed chunk -> 16 kHz mono float32. A chunk starting with a RIFF header is parsed as WAV
    (Sarvam's `audio/wav` messages); anything else is raw PCM in `codec` at `sample_rate`."""
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data)) as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width != 2:
            raise ValueError("WAV chunks must be 16-bit PCM")
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(1)
        return resample(x, sr)
    if codec in ("pcm_s16le", "pcm_l16", "pcm_raw", "wav", "linear16"):
        x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    elif codec == "pcm_f32le":
        x = np.frombuffer(data[: len(data) // 4 * 4], dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported input_audio_codec {codec!r}")
    return resample(x, sample_rate)
