"""In-memory ring buffer logger for /api/logs."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class LogEntry:
    timestamp: float
    level: str
    message: str


class RingLogHandler(logging.Handler):
    def __init__(self, capacity: int = 500):
        super().__init__()
        self._entries: deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        entry = LogEntry(
            timestamp=record.created,
            level=record.levelname.lower(),
            message=self.format(record),
        )
        with self._lock:
            self._entries.append(entry)

    def recent(self, n: int = 100, level: str | None = None) -> list[LogEntry]:
        with self._lock:
            items = list(self._entries)
        if level:
            items = [item for item in items if item.level == level.lower()]
        return items[-n:]


_ring_handler: RingLogHandler | None = None


def setup_logging(level: str = "info", ring_size: int = 500) -> RingLogHandler:
    global _ring_handler
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    _ring_handler = RingLogHandler(capacity=ring_size)
    _ring_handler.setFormatter(formatter)
    root.addHandler(_ring_handler)
    return _ring_handler


def get_ring_handler() -> RingLogHandler:
    if _ring_handler is None:
        return setup_logging()
    return _ring_handler
