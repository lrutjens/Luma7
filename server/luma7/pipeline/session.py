"""Session lifecycle management."""

from __future__ import annotations

import secrets
import threading
from typing import Callable

from luma7.pipeline.types import SessionContext


class SessionManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._active: SessionContext | None = None
        self._sessions: dict[str, SessionContext] = {}

    @property
    def active_session_id(self) -> str | None:
        with self._lock:
            return self._active.session_id if self._active else None

    def is_busy(self) -> bool:
        with self._lock:
            return self._active is not None and not self._active._closed

    def create_session(self) -> SessionContext:
        with self._lock:
            if self._active is not None and not self._active._closed:
                raise RuntimeError("busy")
            session_id = secrets.token_hex(8)
            session = SessionContext(session_id=session_id)
            self._active = session
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> SessionContext | None:
        with self._lock:
            return self._sessions.get(session_id)

    def stop(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.cancel.set()
            return True

    def release(self, session: SessionContext) -> None:
        with self._lock:
            if self._active is session:
                self._active = None
