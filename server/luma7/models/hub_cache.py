"""Download HuggingFace assets once into server/models/hub and load offline."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_HUB_CONFIGURED = False


# Legacy config used the model name without the sentence-transformers org prefix.
_KNOWN_REPO_ALIASES: dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def normalize_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip()
    return _KNOWN_REPO_ALIASES.get(repo_id, repo_id)


def sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", normalize_repo_id(repo_id).strip())


def hub_root(models_root: Path) -> Path:
    return models_root / "hub"


def local_repo_dir(models_root: Path, repo_id: str) -> Path:
    return hub_root(models_root) / sanitize_repo_id(repo_id)


def configure_hub_environment(models_root: Path) -> Path:
    """Point HuggingFace caches at server/models/hub."""
    global _HUB_CONFIGURED
    root = hub_root(models_root)
    root.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(root)
    os.environ["TRANSFORMERS_CACHE"] = str(root)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(root)

    _HUB_CONFIGURED = True
    return root


def set_hub_offline(offline: bool = True) -> None:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def _repo_is_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    markers = ("config.json", "model.safetensors", "weights.safetensors", "pytorch_model.bin")
    return any((path / name).is_file() for name in markers) or any(path.glob("*.safetensors"))


def ensure_hub_repo(models_root: Path, repo_id: str) -> Path:
    """Download a HF repo into models/hub/<repo> if missing."""
    configure_hub_environment(models_root)
    repo_id = normalize_repo_id(repo_id)
    local_dir = local_repo_dir(models_root, repo_id)
    if _repo_is_ready(local_dir):
        logger.info("Using cached hub model %s at %s", repo_id, local_dir)
        return local_dir

    set_hub_offline(False)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    logger.info("Downloading hub model %s -> %s", repo_id, local_dir)
    print(f"Downloading {repo_id} to {local_dir} ...", flush=True)
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    except HfHubHTTPError as exc:
        if exc.response.status_code == 401 and os.environ.get("HF_TOKEN"):
            logger.warning("HF download failed with 401; retrying without HF_TOKEN")
            saved = os.environ.pop("HF_TOKEN", None)
            try:
                snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
            finally:
                if saved:
                    os.environ["HF_TOKEN"] = saved
        else:
            raise
    if not _repo_is_ready(local_dir):
        raise RuntimeError(f"Downloaded hub model incomplete: {local_dir}")
    return local_dir


def ensure_runtime_hub_models(models_root: Path, intent_repo: str, whisper_repo: str) -> None:
    """Ensure HF-backed models exist locally before enabling offline mode."""
    ensure_hub_repo(models_root, intent_repo)
    ensure_hub_repo(models_root, whisper_repo)
    set_hub_offline(True)
    logger.info("Hub cache ready under %s (offline mode enabled)", hub_root(models_root))
