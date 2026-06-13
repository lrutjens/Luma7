"""Tests for OCR section clustering and region resolution."""

from __future__ import annotations

from luma7.pipeline.ocr import OcrLine
from luma7.pipeline.ocr_region import SectionResolver, wants_section_selection
from luma7.pipeline.ocr_sections import cluster_lines_into_sections


def _line(text: str, x: float, y: float, width: float = 0.4, height: float = 0.03) -> OcrLine:
    return OcrLine(text=text, x=x, y=y, width=width, height=height, confidence=0.9)


def test_cluster_headline_and_body_blocks():
    lines = [
        _line("City Council Votes", x=0.1, y=0.05, width=0.8, height=0.05),
        _line("On New Budget Plan", x=0.1, y=0.11, width=0.7, height=0.04),
        _line("The council met Tuesday", x=0.08, y=0.22, width=0.35, height=0.025),
        _line("to approve spending", x=0.08, y=0.26, width=0.33, height=0.025),
        _line("Sports Brief: Tigers Win", x=0.55, y=0.22, width=0.35, height=0.025),
        _line("in overtime thriller", x=0.55, y=0.26, width=0.32, height=0.025),
    ]
    sections = cluster_lines_into_sections(lines)
    assert len(sections) >= 3
    assert sections[0].role == "headline"
    assert "City Council" in sections[0].text


def test_wants_section_selection_detects_spatial_phrases():
    assert wants_section_selection("read the top section")
    assert wants_section_selection("read the left column")
    assert not wants_section_selection("read this")
    assert not wants_section_selection("read everything on the page")


def test_resolver_picks_top_headline():
    lines = [
        _line("Main Headline Here", x=0.1, y=0.05, width=0.8, height=0.05),
        _line("Body copy starts here", x=0.1, y=0.25, width=0.35, height=0.025),
        _line("and continues below", x=0.1, y=0.29, width=0.33, height=0.025),
    ]
    sections = cluster_lines_into_sections(lines)
    resolver = SectionResolver()
    selection = resolver.resolve("read the top section", sections)
    assert selection.mode == "section"
    assert "Main Headline" in selection.text


def test_resolver_reads_all_without_hint():
    lines = [
        _line("Line one", x=0.1, y=0.1),
        _line("Line two", x=0.1, y=0.2),
    ]
    sections = cluster_lines_into_sections(lines)
    resolver = SectionResolver()
    selection = resolver.resolve("read this", sections)
    assert selection.mode == "all"
    assert "Line one" in selection.text
    assert "Line two" in selection.text
