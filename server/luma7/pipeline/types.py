"""Pipeline message types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import queue
import threading
from typing import Callable

import numpy as np


EOS_MARKER = "<EOS>"


class PipelineState(str, Enum):
    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    READING = "reading"
    SPEAKING = "speaking"
    ERROR = "error"
    DONE = "done"


@dataclass
class STTJob:
    session_id: str
    wav_bytes: bytes
    jpeg_bytes: bytes
    text_override: str | None = None


@dataclass
class RoutedJob:
    session_id: str
    transcript: str
    intent: str
    jpeg_bytes: bytes


@dataclass
class TTSJob:
    session_id: str
    text: str
    speed: float | None = None
    trailing_pause_ms: int = 0


@dataclass
class LLMJob:
    session_id: str
    transcript: str
    jpeg_bytes: bytes


@dataclass
class AudioMessage:
    session_id: str
    audio: np.ndarray
    text: str
    is_eos: bool
    sample_rate: int = 22050


@dataclass
class SessionContext:
    session_id: str
    cancel: threading.Event = field(default_factory=threading.Event)
    state: PipelineState = PipelineState.IDLE
    transcript: str = ""
    response_text: str = ""
    sse_queue: queue.Queue[tuple[str, str]] = field(default_factory=queue.Queue)
    _closed: bool = False

    def emit_status(self, state: PipelineState | str) -> None:
        if self._closed or self.cancel.is_set():
            return
        value = state.value if isinstance(state, PipelineState) else state
        self.state = PipelineState(value) if value in PipelineState._value2member_map_ else self.state
        self.sse_queue.put(("status", json.dumps({"state": value})))

    def emit_transcript(self, text: str) -> None:
        if self._closed or self.cancel.is_set():
            return
        self.transcript = text
        self.sse_queue.put(("transcript", json.dumps({"text": text})))

    def emit_intent(self, intent: str, confidence: float, route: str) -> None:
        if self._closed or self.cancel.is_set():
            return
        self.sse_queue.put(
            (
                "intent",
                json.dumps(
                    {
                        "intent": intent,
                        "confidence": round(confidence, 4),
                        "route": route,
                    }
                ),
            )
        )

    def emit_text_chunk(self, text: str, index: int) -> None:
        if self._closed or self.cancel.is_set():
            return
        self.sse_queue.put(("text_chunk", json.dumps({"text": text, "index": index})))

    def emit_error(self, code: str, message: str, recoverable: bool = True) -> None:
        if self._closed:
            return
        self.state = PipelineState.ERROR
        self.sse_queue.put(
            (
                "error",
                json.dumps(
                    {
                        "code": code,
                        "message": message,
                        "recoverable": recoverable,
                    }
                ),
            )
        )

    def emit_audio_chunk(self, wav_b64: str) -> None:
        if self._closed or self.cancel.is_set():
            return
        self.state = PipelineState.SPEAKING
        self.sse_queue.put(("audio_chunk", wav_b64))

    def emit_audio_done(self) -> None:
        if self._closed:
            return
        self.state = PipelineState.DONE
        self.sse_queue.put(("audio_done", "{}"))
        self._closed = True

    def append_response_text(self, text: str) -> None:
        if not text:
            return
        self.response_text = (
            f"{self.response_text} {text}".strip() if self.response_text else text
        )

    def drain_sse(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        while True:
            try:
                items.append(self.sse_queue.get_nowait())
            except queue.Empty:
                break
        return items


StatusCallback = Callable[[SessionContext, PipelineState], None]
