"""Apple Vision OCR via ocrmac."""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrLine:
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0

    @property
    def x_center(self) -> float:
        return self.x + self.width / 2

    @property
    def y_center(self) -> float:
        return self.y + self.height / 2

    @property
    def y_bottom(self) -> float:
        return self.y + self.height

    @property
    def x_right(self) -> float:
        return self.x + self.width


def _parse_annotation(item) -> OcrLine | None:
    if not item:
        return None
    if isinstance(item, str):
        text = item.strip()
        return OcrLine(text=text, x=0.0, y=0.0, width=1.0, height=0.05) if text else None
    if isinstance(item, (list, tuple)):
        if len(item) >= 3 and isinstance(item[0], str):
            text = item[0].strip()
            if not text:
                return None
            confidence = float(item[1]) if len(item) > 1 else 1.0
            bbox = item[2] if len(item) > 2 else (0.0, 0.0, 1.0, 0.05)
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y, width, height = (float(v) for v in bbox[:4])
                return OcrLine(
                    text=text,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                )
        if len(item) >= 1 and isinstance(item[0], str):
            text = item[0].strip()
            return OcrLine(text=text, x=0.0, y=0.0, width=1.0, height=0.05) if text else None
    return None


def run_apple_vision_ocr_lines(jpeg_bytes: bytes) -> tuple[list[OcrLine], float]:
    if not jpeg_bytes:
        return [], 0.0

    from ocrmac import ocrmac

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(jpeg_bytes)
        image_path = Path(tmp.name)

    try:
        t0 = time.perf_counter()
        annotations = ocrmac.OCR(str(image_path)).recognize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        lines: list[OcrLine] = []
        for item in annotations or []:
            parsed = _parse_annotation(item)
            if parsed is not None:
                lines.append(parsed)
        return lines, elapsed_ms
    finally:
        image_path.unlink(missing_ok=True)


def run_apple_vision_ocr(jpeg_bytes: bytes) -> tuple[str, float]:
    lines, elapsed_ms = run_apple_vision_ocr_lines(jpeg_bytes)
    if not lines:
        return "", elapsed_ms
    return "\n".join(line.text for line in lines).strip(), elapsed_ms
