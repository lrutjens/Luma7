"""Ensure Kokoro MLX weights are cached under server/models/hub."""

from __future__ import annotations

import logging
from pathlib import Path

from luma7.config import ServerConfig
from luma7.models.hub_cache import configure_hub_environment, local_repo_dir, set_hub_offline

logger = logging.getLogger(__name__)

DEFAULT_KOKORO_MLX_REPO = "mlx-community/Kokoro-82M-bf16"


def kokoro_mlx_dir(models_root: Path, repo_id: str) -> Path:
    return local_repo_dir(models_root, repo_id)


def _kokoro_mlx_ready(path: Path, voice: str) -> bool:
    if not (path / "config.json").is_file():
        return False
    if not any(p.is_file() and p.parent == path for p in path.glob("*.safetensors")):
        return False
    return (path / "voices" / f"{voice}.safetensors").is_file()


def ensure_kokoro_mlx_model(config: ServerConfig) -> Path:
    """Download Kokoro MLX weights + configured voice into models/hub."""
    configure_hub_environment(config.models_root)
    repo_id = config.kokoro.model.strip()
    voice = config.kokoro.voice.strip()
    local_dir = kokoro_mlx_dir(config.models_root, repo_id)

    if _kokoro_mlx_ready(local_dir, voice):
        logger.info("Using cached Kokoro MLX model at %s", local_dir)
        return local_dir

    from huggingface_hub import snapshot_download

    set_hub_offline(False)
    patterns = [
        "config.json",
        "kokoro-v1_0.safetensors",
        f"voices/{voice}.safetensors",
    ]
    logger.info("Downloading Kokoro MLX %s (voice=%s) -> %s", repo_id, voice, local_dir)
    print(f"Downloading {repo_id} (voice {voice}) to {local_dir} ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=patterns,
    )
    set_hub_offline(True)

    if not _kokoro_mlx_ready(local_dir, voice):
        raise RuntimeError(f"Kokoro MLX model incomplete: {local_dir}")
    return local_dir
