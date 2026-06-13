"""Speech-to-text using mlx-whisper."""

from __future__ import annotations

import logging
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


def wav_bytes_to_audio_array(wav_bytes: bytes):
    import mlx.core as mx

    command = [
        "ffmpeg",
        "-nostdin",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-",
    ]
    process = subprocess.run(
        command,
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace") or "ffmpeg failed")
    audio = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return mx.array(audio)


class SpeechToText:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._loaded = False

    def load(self) -> None:
        from mlx_whisper.load_models import load_model

        logger.info("Loading whisper model from %s", self.model_path)
        load_model(self.model_path, dtype=__import__("mlx.core", fromlist=["core"]).float32)
        self._loaded = True

    def transcribe(self, wav_bytes: bytes) -> str:
        import mlx_whisper

        if not self._loaded:
            self.load()

        audio = wav_bytes_to_audio_array(wav_bytes)
        if int(audio.shape[0]) == 0:
            raise RuntimeError("empty audio")

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            fp16=False,
            verbose=False,
        )
        text = (result.get("text") or "").strip()
        if not text:
            raise RuntimeError("empty transcript")
        return text
