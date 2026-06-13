"""Resolve which OCR section the user wants from their transcript."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from luma7.pipeline.ocr_sections import OcrSection

logger = logging.getLogger(__name__)

_SPATIAL_TOP = re.compile(r"\b(top|upper|headline|title|header|masthead)\b", re.I)
_SPATIAL_BOTTOM = re.compile(r"\b(bottom|lower|footer)\b", re.I)
_SPATIAL_LEFT = re.compile(r"\bleft(\s+(column|side|part|section))?\b", re.I)
_SPATIAL_RIGHT = re.compile(r"\bright(\s+(column|side|part|section))?\b", re.I)
_SPATIAL_ALL = re.compile(r"\b(all|everything|whole|entire|full)\b", re.I)
_SECTION_HINT = re.compile(
    r"\b(section|part|column|story|article|paragraph|caption|blurb|area|region)\b",
    re.I,
)


@dataclass(frozen=True)
class SectionSelection:
    section: OcrSection | None
    mode: str
    score: float
    reason: str

    @property
    def text(self) -> str:
        if self.section is None:
            return ""
        return self.section.text


def wants_section_selection(transcript: str) -> bool:
    text = (transcript or "").strip()
    if not text:
        return False
    if _SPATIAL_ALL.search(text):
        return False
    return bool(
        _SPATIAL_TOP.search(text)
        or _SPATIAL_BOTTOM.search(text)
        or _SPATIAL_LEFT.search(text)
        or _SPATIAL_RIGHT.search(text)
        or _SECTION_HINT.search(text)
    )


class SectionResolver:
    def __init__(self, encoder: SentenceTransformer | None = None):
        self.encoder = encoder

    def resolve(self, transcript: str, sections: list[OcrSection]) -> SectionSelection:
        if not sections:
            return SectionSelection(None, "empty", 0.0, "no sections")

        text = (transcript or "").strip()
        if not text or not wants_section_selection(text):
            merged = _merge_sections(sections)
            return SectionSelection(merged, "all", 1.0, "no section hint in transcript")

        if _SPATIAL_ALL.search(text):
            merged = _merge_sections(sections)
            return SectionSelection(merged, "all", 1.0, "read everything")

        scores = [_score_section(text, section) for section in sections]
        best_index = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_index]
        best = sections[best_index]
        reason = _selection_reason(text, best)
        logger.info(
            "Section resolver picked id=%s role=%s score=%.3f reason=%s",
            best.id,
            best.role,
            best_score,
            reason,
        )
        return SectionSelection(best, "section", best_score, reason)


def _merge_sections(sections: list[OcrSection]) -> OcrSection:
    from luma7.pipeline.ocr import OcrLine

    lines: list[OcrLine] = []
    for section in sorted(sections, key=lambda item: (item.y, item.x)):
        lines.extend(section.lines)
    if not lines:
        return sections[0]
    x0 = min(line.x for line in lines)
    y0 = min(line.y for line in lines)
    x1 = max(line.x_right for line in lines)
    y1 = max(line.y_bottom for line in lines)
    preview = " ".join(line.text for line in lines)
    if len(preview) > 160:
        preview = f"{preview[:159].rstrip()}…"
    return OcrSection(
        id=-1,
        lines=tuple(lines),
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        preview=preview,
        role="all",
    )


def _score_section(transcript: str, section: OcrSection) -> float:
    spatial = _spatial_score(transcript, section)
    semantic = _semantic_score(transcript, section.preview)
    if spatial > 0:
        return 0.65 * spatial + 0.35 * semantic
    return semantic


def _spatial_score(transcript: str, section: OcrSection) -> float:
    score = 0.0
    if _SPATIAL_TOP.search(transcript) or re.search(r"\bheadline\b", transcript, re.I):
        score = max(score, 1.0 - section.y_center)
        if section.role == "headline":
            score += 0.35
        score += min(section.height * 2.0, 0.25)
    if _SPATIAL_BOTTOM.search(transcript):
        score = max(score, section.y_center)
    if _SPATIAL_LEFT.search(transcript):
        if section.x_center < 0.5:
            score = max(score, 1.0 - section.x_center)
    if _SPATIAL_RIGHT.search(transcript):
        if section.x_center > 0.5:
            score = max(score, section.x_center)
    return min(score, 1.5)


def _semantic_score(transcript: str, preview: str) -> float:
    if not preview.strip():
        return 0.0
    # Lightweight keyword overlap when no encoder is available.
    query_tokens = set(re.findall(r"[a-z0-9']+", transcript.lower()))
    preview_tokens = set(re.findall(r"[a-z0-9']+", preview.lower()))
    stop = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "read",
        "me",
        "please",
        "what",
        "does",
        "say",
        "this",
        "that",
        "is",
        "it",
    }
    query_tokens -= stop
    preview_tokens -= stop
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & preview_tokens) / len(query_tokens)
    return overlap


def _selection_reason(transcript: str, section: OcrSection) -> str:
    if _SPATIAL_TOP.search(transcript):
        return f"top/headline match ({section.role})"
    if _SPATIAL_BOTTOM.search(transcript):
        return "bottom match"
    if _SPATIAL_LEFT.search(transcript):
        return "left column match"
    if _SPATIAL_RIGHT.search(transcript):
        return "right column match"
    if _semantic_score(transcript, section.preview) > 0:
        return "topic overlap"
    return f"best match ({section.role})"


class MiniLMSectionResolver(SectionResolver):
    def __init__(self, encoder: SentenceTransformer):
        super().__init__(encoder)

    def resolve(self, transcript: str, sections: list[OcrSection]) -> SectionSelection:
        if not sections:
            return SectionSelection(None, "empty", 0.0, "no sections")

        text = (transcript or "").strip()
        if not text or not wants_section_selection(text):
            merged = _merge_sections(sections)
            return SectionSelection(merged, "all", 1.0, "no section hint in transcript")

        if _SPATIAL_ALL.search(text):
            merged = _merge_sections(sections)
            return SectionSelection(merged, "all", 1.0, "read everything")

        assert self.encoder is not None
        query_embedding = self.encoder.encode([text])
        previews = [section.preview or section.text for section in sections]
        preview_embeddings = self.encoder.encode(previews)

        scores: list[float] = []
        for index, section in enumerate(sections):
            semantic = float(cos_sim(query_embedding, preview_embeddings[index : index + 1])[0][0])
            spatial = _spatial_score(text, section)
            if spatial > 0:
                scores.append(0.55 * spatial + 0.45 * semantic)
            else:
                scores.append(semantic)

        best_index = max(range(len(scores)), key=lambda i: scores[i])
        best = sections[best_index]
        return SectionSelection(
            best,
            "section",
            scores[best_index],
            _selection_reason(text, best),
        )
