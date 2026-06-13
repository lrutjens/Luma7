"""Sentence boundary detection and OCR chunk splitting."""

from __future__ import annotations

import re

EOS_MARKER = "<EOS>"
MIN_WORDS = 4

_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
    "st",
    "ave",
    "inc",
    "ltd",
}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _is_decimal_boundary(text: str, idx: int) -> bool:
    if idx <= 0 or idx >= len(text):
        return False
    if text[idx] != ".":
        return False
    left = text[:idx]
    right = text[idx + 1 :]
    return bool(re.search(r"\d$", left)) and bool(re.match(r"\d", right))


def _is_abbreviation_boundary(text: str, idx: int) -> bool:
    if text[idx] != ".":
        return False
    window = text[max(0, idx - 8) : idx].lower().strip()
    token = window.split()[-1] if window.split() else window
    return token.rstrip(".") in _ABBREVIATIONS or token in _ABBREVIATIONS


def _is_sentence_end(text: str, idx: int) -> bool:
    ch = text[idx]
    if ch in "!?":
        return True
    if ch != ".":
        return False
    return not _is_decimal_boundary(text, idx) and not _is_abbreviation_boundary(text, idx)


def normalize_ocr_text(text: str) -> str:
    """Collapse OCR whitespace/newlines into one paragraph with single spaces."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # Line-wrapped hyphenations: "exam-\nple" -> "example"
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    # Leading dash/bullet lines from OCR layout
    cleaned = re.sub(r"(?m)^-\s+", "", cleaned)
    cleaned = cleaned.replace("- ", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def group_ocr_paragraphs(sentences: list[str], max_chars: int = 480) -> list[str]:
    """Group sentences into longer narration chunks for smoother audiobook flow."""
    if not sentences:
        return []

    groups: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            groups.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += extra

    if current:
        groups.append(" ".join(current))
    return groups


def split_on_sentence_boundaries(text: str) -> list[str]:
    """Split normalized text on . ! ? while respecting abbreviations and decimals."""
    body = normalize_ocr_text(text)
    if not body:
        return []

    sentences: list[str] = []
    start = 0
    for idx, _ch in enumerate(body):
        if not _is_sentence_end(body, idx):
            continue
        candidate = body[start : idx + 1].strip()
        if candidate:
            sentences.append(candidate)
        start = idx + 1

    remainder = body[start:].strip()
    if remainder:
        sentences.append(remainder)
    return sentences


class SentenceBuffer:
    """Accumulate streamed tokens and emit speakable sentence chunks."""

    def __init__(self, min_words: int = MIN_WORDS):
        self._buffer = ""
        self._min_words = min_words
        self._in_thinking = False

    def feed(self, token: str) -> list[str]:
        if not token:
            return []

        if "<|channel>thought" in token:
            self._in_thinking = True
            return []

        if self._in_thinking:
            if "<channel|>" in token:
                self._in_thinking = False
            return []

        self._buffer += token
        return self._drain_complete_sentences()

    def flush(self) -> list[str]:
        text = self._buffer.strip()
        self._buffer = ""
        if not text or _word_count(text) < 1:
            return []
        return [text]

    def _drain_complete_sentences(self) -> list[str]:
        sentences: list[str] = []
        while True:
            match = re.search(r"([.!?])(\s+)", self._buffer)
            if not match:
                break
            punct_idx = match.start(1)
            if _is_decimal_boundary(self._buffer, punct_idx) or _is_abbreviation_boundary(
                self._buffer, punct_idx
            ):
                self._buffer = self._buffer[: punct_idx + 1] + self._buffer[punct_idx + 1 :]
                continue

            candidate = self._buffer[: match.end(1)].strip()
            remainder = self._buffer[match.end() :]
            if _word_count(candidate) < self._min_words:
                break

            sentences.append(candidate)
            self._buffer = remainder

        return sentences


def split_ocr_text(text: str, max_paragraph_chars: int = 480) -> list[str]:
    sentences = split_on_sentence_boundaries(text)
    paragraphs = group_ocr_paragraphs(sentences, max_chars=max_paragraph_chars)
    if paragraphs:
        return paragraphs
    return ["I couldn't read any text in the image."]
