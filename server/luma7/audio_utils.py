"""Audio encoding helpers for SSE output."""

from __future__ import annotations

import base64
import io
import struct
import wave

import numpy as np
from scipy import signal


def float32_to_pcm16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.float32)
    count = int(round(len(samples) * dst_rate / src_rate))
    if count <= 0:
        return np.array([], dtype=np.float32)
    return signal.resample(samples, count).astype(np.float32)


def pcm16_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buffer.getvalue()


def float32_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    return pcm16_to_wav_bytes(float32_to_pcm16(samples), sample_rate)


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("ascii")


def strip_wav_header(wav_bytes: bytes) -> bytes:
    if len(wav_bytes) <= 44:
        return b""
    return wav_bytes[44:]


def parse_wav_pcm16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16)
    return pcm, sample_rate
