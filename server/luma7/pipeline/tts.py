"""Kokoro text-to-speech via mlx-audio on Apple Silicon."""

from __future__ import annotations

import logging
import re

import numpy as np

from luma7.audio_utils import resample
from luma7.pipeline.kokoro_mlx_patch import apply_kokoro_mlx_patch, harden_kokoro_g2p
from luma7.pipeline.tts_text import sanitize_for_tts

logger = logging.getLogger(__name__)

apply_kokoro_mlx_patch()

# mlx-audio KokoroPipeline language aliases (en-us -> American English)
_LANG_TO_CODE = {
    "en": "a",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "pt": "p",
    "ja": "j",
    "zh": "z",
}


def _lang_code(lang: str) -> str:
    return _LANG_TO_CODE.get(lang.strip().lower(), lang.strip().lower() or "a")


class TextToSpeech:
    def __init__(
        self,
        model_path: str,
        voice: str,
        speed: float,
        lang: str,
        output_sample_rate: int,
    ):
        self.model_path = model_path
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.lang_code = _lang_code(lang)
        self.output_sample_rate = output_sample_rate
        self._model = None
        self._native_rate: int | None = None

    def load(self) -> None:
        from mlx_audio.tts.utils import load

        logger.info("Loading Kokoro MLX TTS from %s", self.model_path)
        self._model = load(self.model_path, lazy=False)
        self._native_rate = int(self._model.sample_rate)

        for _result in self._model.generate(
            "warmup",
            voice=self.voice,
            speed=self.speed,
            lang_code=self.lang_code,
            split_pattern=r"(?!)",
        ):
            break

        harden_kokoro_g2p(self._model)

        logger.info("Kokoro MLX ready (native rate %s Hz)", self._native_rate)

    def synthesize(
        self,
        text: str,
        *,
        speed: float | None = None,
        trailing_pause_ms: int = 0,
    ) -> tuple[np.ndarray, int]:
        if self._model is None:
            raise RuntimeError("TTS not loaded")
        cleaned = sanitize_for_tts(text)
        if not cleaned or cleaned == "<EOS>":
            return np.array([], dtype=np.float32), self.output_sample_rate

        speech_speed = self.speed if speed is None else speed
        audio_parts: list[np.ndarray] = []
        native = self._native_rate or 24000

        try:
            results = self._model.generate(
                cleaned,
                voice=self.voice,
                speed=speech_speed,
                lang_code=self.lang_code,
                split_pattern=r"(?!)",
            )
        except TypeError as exc:
            if "NoneType" not in str(exc):
                raise
            logger.warning("G2P failed for %r; retrying with ASCII-only text", cleaned[:120])
            cleaned = re.sub(r"[^A-Za-z0-9 .,!?';:-]", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                return np.array([], dtype=np.float32), self.output_sample_rate
            results = self._model.generate(
                cleaned,
                voice=self.voice,
                speed=speech_speed,
                lang_code=self.lang_code,
                split_pattern=r"(?!)",
            )

        for result in results:
            segment = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            if segment.size:
                audio_parts.append(segment)
            native = int(result.sample_rate)

        if not audio_parts:
            return np.array([], dtype=np.float32), self.output_sample_rate

        audio = np.concatenate(audio_parts)
        if trailing_pause_ms > 0:
            pause_samples = int(round(native * trailing_pause_ms / 1000))
            if pause_samples > 0:
                audio = np.concatenate([audio, np.zeros(pause_samples, dtype=np.float32)])

        if native != self.output_sample_rate:
            audio = resample(audio, native, self.output_sample_rate)
        return audio, self.output_sample_rate
