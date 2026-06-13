"""Cluster OCR lines into layout sections with bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass

from luma7.pipeline.ocr import OcrLine


@dataclass(frozen=True)
class OcrSection:
    id: int
    lines: tuple[OcrLine, ...]
    x: float
    y: float
    width: float
    height: float
    preview: str
    role: str

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines).strip()

    @property
    def x_center(self) -> float:
        return self.x + self.width / 2

    @property
    def y_center(self) -> float:
        return self.y + self.height / 2

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bbox": {
                "x": round(self.x, 4),
                "y": round(self.y, 4),
                "width": round(self.width, 4),
                "height": round(self.height, 4),
            },
            "preview": self.preview,
            "role": self.role,
            "line_count": len(self.lines),
        }


def _union_bbox(lines: list[OcrLine]) -> tuple[float, float, float, float]:
    x0 = min(line.x for line in lines)
    y0 = min(line.y for line in lines)
    x1 = max(line.x_right for line in lines)
    y1 = max(line.y_bottom for line in lines)
    return x0, y0, x1 - x0, y1 - y0


def _preview_text(lines: list[OcrLine], limit: int = 160) -> str:
    text = " ".join(line.text for line in lines).strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def _infer_role(
    section_id: int,
    y: float,
    height: float,
    line_count: int,
    median_height: float,
    top_y: float,
) -> str:
    if section_id == 0 and y <= top_y + 0.02 and height >= median_height * 1.2 and line_count <= 3:
        return "headline"
    if y < 0.33:
        return "top"
    if y + height > 0.67:
        return "bottom"
    if line_count >= 4:
        return "body"
    return "block"


def cluster_lines_into_sections(lines: list[OcrLine]) -> list[OcrSection]:
    if not lines:
        return []

    sorted_lines = sorted(lines, key=lambda line: (line.y, line.x))
    heights = sorted(line.height for line in lines)
    median_height = heights[len(heights) // 2]
    y_gap = max(0.012, median_height * 1.5)
    x_align_tol = max(0.05, median_height * 2.5)

    groups: list[list[OcrLine]] = []
    current = [sorted_lines[0]]
    for line in sorted_lines[1:]:
        prev = current[-1]
        vertical_gap = line.y - prev.y_bottom
        x_aligned = abs(line.x - prev.x) <= x_align_tol
        horizontal_overlap = line.x < prev.x_right and line.x_right > prev.x
        if vertical_gap <= y_gap and (x_aligned or horizontal_overlap):
            current.append(line)
        else:
            groups.append(current)
            current = [line]
    groups.append(current)

    top_y = min(line.y for line in lines)
    sections: list[OcrSection] = []
    for index, group in enumerate(groups):
        x, y, width, height = _union_bbox(group)
        preview = _preview_text(group)
        role = _infer_role(index, y, height, len(group), median_height, top_y)
        sections.append(
            OcrSection(
                id=index,
                lines=tuple(group),
                x=x,
                y=y,
                width=width,
                height=height,
                preview=preview,
                role=role,
            )
        )
    return sections
