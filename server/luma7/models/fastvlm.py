"""Ensure FastVLM MLX weights exist locally before loading."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from luma7.config import ServerConfig

logger = logging.getLogger(__name__)

APPLE_BASE_URL = "https://ml-site.cdn-apple.com/datasets/fastvlm"
MIN_FREE_BYTES_FOR_7B = 22 * 1024 * 1024 * 1024  # ~22 GB for extract + convert (zip deleted after extract)

CHECKPOINT_URLS: dict[str, str] = {
    "llava-fastvithd_0.5b_stage3": f"{APPLE_BASE_URL}/llava-fastvithd_0.5b_stage3.zip",
    "llava-fastvithd_1.5b_stage3": f"{APPLE_BASE_URL}/llava-fastvithd_1.5b_stage3.zip",
    "llava-fastvithd_7b_stage3": f"{APPLE_BASE_URL}/llava-fastvithd_7b_stage3.zip",
}


class FastVLMModelError(RuntimeError):
    pass


def _safetensors_keys(path: Path) -> list[str]:
    from safetensors import safe_open

    weights = path / "model.safetensors"
    if not weights.is_file():
        return []
    with safe_open(str(weights), framework="numpy") as handle:
        return list(handle.keys())


def _is_mlx_vlm_compatible(path: Path) -> bool:
    """True when weights match mlx-vlm FastVLM layout (not the iOS-only Apple bundle)."""
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    keys = _safetensors_keys(path)
    if not keys:
        return False
    if any(key.startswith("multi_modal_projector") for key in keys):
        return False
    if not any(key.startswith("mm_projector") for key in keys):
        return False
    if not any("vision_tower" in key for key in keys):
        return False
    return True


def _download_with_progress(url: str, dest: Path) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    print(f"Downloading {url} ...", flush=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        if block_num % 200 == 0:
            msg = f"Download progress: {pct}%"
            logger.info(msg)
            print(msg, flush=True)

    urlretrieve(url, dest, _report)


def _extract_checkpoint_zip(zip_path: Path, checkpoint_name: str, dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dest_root)

    extracted = dest_root / checkpoint_name
    if extracted.is_dir() and (extracted / "config.json").is_file():
        return extracted

    candidates = [
        path
        for path in dest_root.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise FastVLMModelError(f"Could not find extracted checkpoint for {checkpoint_name}")


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return usage.free


def _require_disk_space(path: Path, checkpoint: str) -> None:
    if "7b" not in checkpoint.lower():
        return
    free = _free_bytes(path)
    if free < MIN_FREE_BYTES_FOR_7B:
        needed_gb = MIN_FREE_BYTES_FOR_7B / (1024**3)
        free_gb = free / (1024**3)
        raise FastVLMModelError(
            f"Not enough disk space for {checkpoint}. "
            f"Need ~{needed_gb:.0f} GB free, have {free_gb:.1f} GB. "
            "Free space or use fastvlm.checkpoint: llava-fastvithd_1.5b_stage3 in config.yaml."
        )


def _download_checkpoint(checkpoint: str, checkpoint_dest: Path) -> Path:
    url = CHECKPOINT_URLS.get(checkpoint)
    if url is None:
        raise FastVLMModelError(f"No download URL configured for checkpoint '{checkpoint}'")

    if checkpoint_dest.is_dir() and (checkpoint_dest / "config.json").is_file():
        return checkpoint_dest

    checkpoint_dest.parent.mkdir(parents=True, exist_ok=True)
    _require_disk_space(checkpoint_dest.parent, checkpoint)

    zip_path = checkpoint_dest.parent / f"{checkpoint}.zip"
    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000_000:
        if zip_path.is_file():
            zip_path.unlink()
        _download_with_progress(url, zip_path)

    staging = checkpoint_dest.parent / f".{checkpoint}.extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    extracted = _extract_checkpoint_zip(zip_path, checkpoint, staging)
    if checkpoint_dest.exists():
        shutil.rmtree(checkpoint_dest)
    shutil.move(str(extracted), str(checkpoint_dest))
    shutil.rmtree(staging, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    return checkpoint_dest


def _patch_checkpoint_config(checkpoint_dir: Path) -> None:
    """Fix known FastVLM config issues before mlx-vlm conversion."""
    import json

    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text())
    if config.get("tie_word_embeddings") is True and config.get("architectures") == ["LlavaQwen2ForCausalLM"]:
        config["tie_word_embeddings"] = False
        config_path.write_text(json.dumps(config, indent=2))
        logger.info("Patched tie_word_embeddings=false in %s", config_path)


def _convert_from_checkpoint(
    checkpoint_dir: Path,
    mlx_dest: Path,
    quantize_bits: int | None,
) -> None:
    _patch_checkpoint_config(checkpoint_dir)
    if mlx_dest.exists():
        shutil.rmtree(mlx_dest)
    mlx_dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm",
        "convert",
        "--hf-path",
        str(checkpoint_dir),
        "--mlx-path",
        str(mlx_dest),
    ]
    if quantize_bits:
        cmd.extend(["-q", "--q-bits", str(quantize_bits)])
    logger.info("Converting checkpoint to MLX: %s", " ".join(cmd))
    print(f"Converting {checkpoint_dir.name} to MLX (this may take several minutes)...", flush=True)
    subprocess.run(cmd, check=True)


def _resolve_checkpoint_dir(config: ServerConfig) -> Path | None:
    checkpoint = config.fastvlm.checkpoint
    candidates = [
        config.fastvlm_checkpoint_path,
        config.models_root.parent.parent / "ml-fastvlm" / "checkpoints" / checkpoint,
    ]
    for path in candidates:
        if path.is_dir() and (path / "config.json").is_file():
            return path
    return None


def ensure_fastvlm_mlx_model(config: ServerConfig) -> Path:
    """Return a mlx-vlm-compatible MLX model directory, downloading/converting if needed."""
    from luma7.models.hub_cache import set_hub_offline

    mlx_path = config.fastvlm_mlx_path
    if _is_mlx_vlm_compatible(mlx_path):
        return mlx_path

    set_hub_offline(False)

    if mlx_path.exists():
        logger.warning("Removing incompatible MLX bundle at %s", mlx_path)
        shutil.rmtree(mlx_path)

    checkpoint = config.fastvlm.checkpoint
    checkpoint_dir = _resolve_checkpoint_dir(config)
    if checkpoint_dir is None:
        checkpoint_dir = config.fastvlm_checkpoint_path
        logger.info("Downloading PyTorch checkpoint %s", checkpoint)
        print(f"Preparing {checkpoint} (download + mlx-vlm convert)...", flush=True)
        _download_checkpoint(checkpoint, checkpoint_dir)

    _convert_from_checkpoint(checkpoint_dir, mlx_path, config.fastvlm.quantize_bits)
    if not _is_mlx_vlm_compatible(mlx_path):
        raise FastVLMModelError(
            f"Converted MLX model at {mlx_path} is still incompatible with mlx-vlm."
        )

    logger.info("FastVLM MLX ready at %s", mlx_path)
    return mlx_path
