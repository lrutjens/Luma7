"""Normalize text before Kokoro TTS."""

from __future__ import annotations

import re
import unicodedata


def sanitize_for_tts(text: str) -> str:
    """Make OCR/LLM text safer for misaki G2P."""
    normalized = unicodedata.normalize("NFKC", text or "")
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2022": ". ",
        "\u2026": ".",
        "\u00a0": " ",
        "\u00ad": "",
        "\ufeff": "",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)

    # Drop control chars and symbols that commonly break G2P.
    normalized = "".join(
        ch
        for ch in normalized
        if ch in "\n\t\r" or unicodedata.category(ch)[0] not in ("C",)
    )
    normalized = re.sub(r"[^\w\s.,!?;:'\"()/-]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
