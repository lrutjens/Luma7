#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
FASTVLM_REF="$(cd "$ROOT/../ml-fastvlm" 2>/dev/null && pwd || true)"

mkdir -p "$MODELS/kokoro" "$MODELS/mlx" "$MODELS/checkpoints"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null; then
  PY="python3"
else
  echo "python3 not found" >&2
  exit 1
fi

echo "==> Caching Kokoro MLX TTS model"
"$PY" -c "
from luma7.config import load_config
from luma7.models.hub_cache import configure_hub_environment
from luma7.models.kokoro_mlx import ensure_kokoro_mlx_model
cfg = load_config()
configure_hub_environment(cfg.models_root)
print(ensure_kokoro_mlx_model(cfg))
"

echo "==> Seeding FastVLM checkpoints from ml-fastvlm (if present)"
if [[ -n "$FASTVLM_REF" ]]; then
  for ckpt in llava-fastvithd_1.5b_stage3 llava-fastvithd_7b_stage3; do
    if [[ -d "$FASTVLM_REF/checkpoints/$ckpt" && ! -e "$MODELS/checkpoints/$ckpt" ]]; then
      echo "Linking checkpoint $ckpt"
      ln -sf "$FASTVLM_REF/checkpoints/$ckpt" "$MODELS/checkpoints/$ckpt"
    fi
    if [[ -d "$FASTVLM_REF/mlx_models/$ckpt" && ! -e "$MODELS/mlx/$ckpt" ]]; then
      compatible="$("$ROOT/.venv/bin/python" -c "
from pathlib import Path
from luma7.models.fastvlm import _is_mlx_vlm_compatible
print('yes' if _is_mlx_vlm_compatible(Path('$FASTVLM_REF/mlx_models/$ckpt')) else 'no')
" 2>/dev/null || echo no)"
      if [[ "$compatible" == "yes" ]]; then
        echo "Linking MLX weights $ckpt"
        ln -sf "$FASTVLM_REF/mlx_models/$ckpt" "$MODELS/mlx/$ckpt"
      fi
    fi
  done
fi

echo "==> Caching HuggingFace models locally (intent + whisper)"
"$PY" -c "
from luma7.config import load_config
from luma7.models.hub_cache import configure_hub_environment, ensure_runtime_hub_models
cfg = load_config()
configure_hub_environment(cfg.models_root)
ensure_runtime_hub_models(cfg.models_root, cfg.intent.encoder, cfg.whisper.model)
print('Hub cache:', cfg.models_root / 'hub')
"

echo "==> Ensuring configured FastVLM MLX model (download + convert if needed)"
"$PY" -c "from luma7.config import load_config; from luma7.models.fastvlm import ensure_fastvlm_mlx_model; print(ensure_fastvlm_mlx_model(load_config()))"

echo "==> Training intent classifier"
"$PY" "$ROOT/scripts/train_intent.py"

echo "==> Setup complete"
echo "Models directory: $MODELS"
echo "Start server: cd $ROOT && python -m luma7"
