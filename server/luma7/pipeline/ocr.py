"""Apple Vision OCR via ocrmac."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_apple_vision_ocr(jpeg_bytes: bytes) -> tuple[str, float]:
    if not jpeg_bytes:
        return "", 0.0

    from ocrmac import ocrmac

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(jpeg_bytes)
        image_path = Path(tmp.name)

    try:
        t0 = time.perf_counter()
        annotations = ocrmac.OCR(str(image_path)).recognize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if not annotations:
            return "", elapsed_ms
        lines = [item[0] for item in annotations if item and item[0]]
        return "\n".join(lines).strip(), elapsed_ms
    finally:
        image_path.unlink(missing_ok=True)
