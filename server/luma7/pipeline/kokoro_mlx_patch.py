"""Runtime patch for mlx-audio v0.4.4 Kokoro interpolate float-precision bug."""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx

logger = logging.getLogger(__name__)
_PATCHED = False


def _stable_ceil_size(length: int, scale: float) -> int:
    val = float(length) * float(scale)
    rounded = round(val, 9)
    if abs(val - rounded) < 1e-9:
        val = rounded
    return max(1, int(math.ceil(val)))


def apply_kokoro_mlx_patch() -> None:
    """Fix broadcast shape mismatches in Kokoro TTS (mlx-audio #786)."""
    global _PATCHED
    if _PATCHED:
        return

    from mlx_audio.tts.models import interpolate as interpolate_mod

    def patched_interpolate(
        input: mx.array,
        size: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        scale_factor: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
    ) -> mx.array:
        ndim = input.ndim
        if ndim < 3:
            raise ValueError(f"Expected at least 3D input (N, C, D1), got {ndim}D")

        spatial_dims = ndim - 2

        if size is not None and scale_factor is not None:
            raise ValueError("Only one of size or scale_factor should be defined")
        if size is None and scale_factor is None:
            raise ValueError("One of size or scale_factor must be defined")

        if size is not None and not isinstance(size, (list, tuple)):
            size = [size] * spatial_dims
        if scale_factor is not None and not isinstance(scale_factor, (list, tuple)):
            scale_factor = [scale_factor] * spatial_dims

        if size is None:
            size = [
                _stable_ceil_size(int(input.shape[i + 2]), float(scale_factor[i]))
                for i in range(spatial_dims)
            ]

        if spatial_dims == 1:
            return interpolate_mod.interpolate1d(input, size[0], mode, align_corners)
        raise ValueError(f"Only 1D interpolation currently supported, got {spatial_dims}D")

    interpolate_mod.interpolate = patched_interpolate
    try:
        from mlx_audio.tts.models.kokoro import istftnet

        istftnet.interpolate = patched_interpolate
    except ImportError:
        pass
    _PATCHED = True
    logger.debug("Applied Kokoro MLX interpolate patch")


def _silent_g2p_fallback(token) -> tuple[str, int]:
    """Skip tokens misaki cannot pronounce when espeak is unavailable."""
    return "", 3


def configure_espeak_library() -> bool:
    """Point phonemizer at a Homebrew espeak-ng install if present."""
    import glob
    import os
    import platform

    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        return False

    if EspeakWrapper._ESPEAK_LIBRARY:
        return True

    if platform.system() != "Darwin":
        return False

    patterns = (
        "/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng*.dylib",
        "/usr/local/Cellar/espeak-ng/*/lib/libespeak-ng*.dylib",
    )
    for pattern in patterns:
        for library in sorted(glob.glob(pattern), reverse=True):
            if os.path.exists(library):
                EspeakWrapper.set_library(library)
                logger.info("Using espeak-ng library at %s", library)
                return True
    return False


def harden_kokoro_g2p(model) -> None:
    """Ensure unknown words do not crash misaki when espeak is missing."""
    configure_espeak_library()
    pipelines = getattr(model, "_pipelines", None) or {}
    for pipeline in pipelines.values():
        g2p = getattr(pipeline, "g2p", None)
        if g2p is None or getattr(g2p, "fallback", None) is not None:
            continue
        try:
            from misaki import espeak

            british = getattr(g2p, "british", False)
            g2p.fallback = espeak.EspeakFallback(british=british)
            logger.info("Kokoro espeak G2P fallback enabled")
        except Exception as exc:
            g2p.fallback = _silent_g2p_fallback
            logger.warning(
                "espeak-ng unavailable (%s); unpronounceable tokens will be skipped",
                exc,
            )
