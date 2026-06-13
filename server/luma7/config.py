"""Server configuration loading and validation."""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SERVER_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FastVLMConfig:
    checkpoint: str = "llava-fastvithd_7b_stage3"
    quantize_bits: int | None = 4
    system_prompt: str = "You are a helpful vision assistant for smart glasses."
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256


@dataclass
class WhisperConfig:
    model: str = "mlx-community/whisper-small-mlx"


@dataclass
class IntentConfig:
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier_path: str = "models/intent_classifier.joblib"


@dataclass
class KokoroConfig:
    model: str = "mlx-community/Kokoro-82M-bf16"
    voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"
    output_sample_rate: int = 24000


@dataclass
class OCRConfig:
    no_image_message: str = "There's no image to read."
    audiobook_speed: float = 0.9
    paragraph_pause_ms: int = 500
    max_paragraph_chars: int = 480
    section_selection: bool = True


@dataclass
class TTSJob:
    session_id: str
    text: str
    speed: float | None = None
    trailing_pause_ms: int = 0


@dataclass
class LoggingConfig:
    level: str = "info"
    ring_size: int = 500


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    auth_token: str = ""
    models_dir: str = "models"
    fastvlm: FastVLMConfig = field(default_factory=FastVLMConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    kokoro: KokoroConfig = field(default_factory=KokoroConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: Path = field(default_factory=lambda: SERVER_ROOT / "config.yaml")

    @property
    def models_root(self) -> Path:
        path = Path(self.models_dir)
        return path if path.is_absolute() else SERVER_ROOT / path

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return SERVER_ROOT / path

    @property
    def intent_classifier_path(self) -> Path:
        return self.resolve(self.intent.classifier_path)

    @property
    def kokoro_mlx_path(self) -> Path:
        from luma7.models.kokoro_mlx import kokoro_mlx_dir

        return kokoro_mlx_dir(self.models_root, self.kokoro.model)

    @property
    def fastvlm_mlx_path(self) -> Path:
        return self.models_root / "mlx" / self.fastvlm.checkpoint

    @property
    def fastvlm_checkpoint_path(self) -> Path:
        return self.models_root / "checkpoints" / self.fastvlm.checkpoint

    @property
    def intent_encoder_path(self) -> Path:
        from luma7.models.hub_cache import local_repo_dir

        return local_repo_dir(self.models_root, self.intent.encoder)

    @property
    def whisper_model_path(self) -> Path:
        from luma7.models.hub_cache import local_repo_dir

        return local_repo_dir(self.models_root, self.whisper.model)


def _coerce_token(value: str | None) -> str:
    token = (value or "").strip()
    if len(token) == 32 and token.isalnum():
        return token
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(32))


def _merge_dataclass(instance: Any, data: dict[str, Any]) -> Any:
    for key, value in data.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance


def load_config(path: Path | None = None) -> ServerConfig:
    config_path = path or (SERVER_ROOT / "config.yaml")
    defaults_path = SERVER_ROOT / "config.default.yaml"

    raw: dict[str, Any] = {}
    if defaults_path.is_file():
        raw = yaml.safe_load(defaults_path.read_text()) or {}
    if config_path.is_file():
        user = yaml.safe_load(config_path.read_text()) or {}
        raw.update(user)

    config = ServerConfig(config_path=config_path)
    _merge_dataclass(config, raw)
    config.auth_token = _coerce_token(config.auth_token)
    return config


def save_config(config: ServerConfig) -> None:
    payload = {
        "host": config.host,
        "port": config.port,
        "auth_token": config.auth_token,
        "models_dir": config.models_dir,
        "fastvlm": config.fastvlm.__dict__,
        "whisper": config.whisper.__dict__,
        "intent": config.intent.__dict__,
        "kokoro": config.kokoro.__dict__,
        "ocr": config.ocr.__dict__,
        "logging": config.logging.__dict__,
    }
    config.config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
