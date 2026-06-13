"""Four-thread GLaDOS-style pipeline engine."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from luma7.audio_utils import float32_to_wav_bytes, wav_bytes_to_base64
from luma7.pipeline.ocr import run_apple_vision_ocr_lines
from luma7.pipeline.ocr_region import MiniLMSectionResolver
from luma7.pipeline.ocr_sections import cluster_lines_into_sections
from luma7.pipeline.sentence import EOS_MARKER, normalize_ocr_text, split_ocr_text
from luma7.pipeline.types import (
    AudioMessage,
    LLMJob,
    PipelineState,
    RoutedJob,
    STTJob,
    TTSJob,
)

if TYPE_CHECKING:
    from luma7.config import ServerConfig
    from luma7.pipeline.intent import IntentClassifier
    from luma7.pipeline.llm import FastVLMEngine
    from luma7.pipeline.session import SessionManager
    from luma7.pipeline.stt import SpeechToText
    from luma7.pipeline.tts import TextToSpeech

logger = logging.getLogger(__name__)


def _drain_queue(q: queue.Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


class PipelineEngine:
    def __init__(
        self,
        config: ServerConfig,
        sessions: SessionManager,
        stt: SpeechToText,
        intent: IntentClassifier,
        llm: FastVLMEngine,
        tts: TextToSpeech,
    ):
        self.config = config
        self.sessions = sessions
        self.stt = stt
        self.intent = intent
        self.llm = llm
        self.tts = tts

        self.stt_queue: queue.Queue[STTJob | None] = queue.Queue()
        self.llm_queue: queue.Queue[LLMJob | None] = queue.Queue()
        self.tts_queue: queue.Queue[TTSJob | None] = queue.Queue()
        self.audio_queue: queue.Queue[AudioMessage | None] = queue.Queue()

        self._text_chunk_counters: dict[str, int] = {}
        self._section_resolver: MiniLMSectionResolver | None = None
        self._threads: list[threading.Thread] = []
        self._started = False
        self._models_ready = False

    @property
    def models_ready(self) -> bool:
        return self._models_ready

    def load_models(self) -> None:
        from luma7.models.fastvlm import ensure_fastvlm_mlx_model
        from luma7.models.hub_cache import configure_hub_environment, ensure_runtime_hub_models, set_hub_offline
        from luma7.models.kokoro_mlx import ensure_kokoro_mlx_model

        configure_hub_environment(self.config.models_root)
        ensure_runtime_hub_models(
            self.config.models_root,
            self.config.intent.encoder,
            self.config.whisper.model,
        )

        self.intent.load()
        assert self.intent.encoder is not None
        self._section_resolver = MiniLMSectionResolver(self.intent.encoder)
        self.stt.load()
        mlx_path = ensure_fastvlm_mlx_model(self.config)
        self.llm.model_path = mlx_path
        self.llm.load()
        kokoro_path = ensure_kokoro_mlx_model(self.config)
        self.tts.model_path = str(kokoro_path)
        self.tts.load()
        set_hub_offline(True)
        self._models_ready = True
        logger.info("All models loaded")

    def start(self) -> None:
        if self._started:
            return
        workers = [
            ("stt", self._stt_worker),
            ("llm", self._llm_worker),
            ("tts", self._tts_worker),
            ("sse", self._sse_worker),
        ]
        for name, target in workers:
            thread = threading.Thread(target=target, name=f"pipeline-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        self._started = True
        logger.info("Pipeline threads started")

    def submit(self, job: STTJob) -> None:
        self.stt_queue.put(job)

    def _session_for(self, session_id: str):
        from luma7.pipeline.types import SessionContext

        session = self.sessions.get(session_id)
        if session is None:
            raise RuntimeError(f"unknown session {session_id}")
        return session

    def _select_ocr_text(self, session, transcript: str, lines) -> str:
        if not lines:
            return ""
        if not self.config.ocr.section_selection or self._section_resolver is None:
            return "\n".join(line.text for line in lines).strip()

        sections = cluster_lines_into_sections(lines)
        session.emit_ocr_sections([section.to_dict() for section in sections])
        selection = self._section_resolver.resolve(transcript, sections)
        if selection.section is not None:
            session.emit_ocr_section_selected(
                {
                    "section_id": selection.section.id,
                    "mode": selection.mode,
                    "score": round(selection.score, 4),
                    "reason": selection.reason,
                    "role": selection.section.role,
                    "bbox": {
                        "x": round(selection.section.x, 4),
                        "y": round(selection.section.y, 4),
                        "width": round(selection.section.width, 4),
                        "height": round(selection.section.height, 4),
                    },
                }
            )
        return selection.text

    def _enqueue_tts(
        self,
        session_id: str,
        text: str,
        *,
        speed: float | None = None,
        trailing_pause_ms: int = 0,
    ) -> None:
        if text:
            self.tts_queue.put(
                TTSJob(
                    session_id=session_id,
                    text=text,
                    speed=speed,
                    trailing_pause_ms=trailing_pause_ms,
                )
            )

    def _stt_worker(self) -> None:
        while True:
            job = self.stt_queue.get()
            if job is None:
                return
            session = self._session_for(job.session_id)
            try:
                if session.cancel.is_set():
                    continue

                session.emit_status(PipelineState.TRANSCRIBING)
                if job.text_override:
                    transcript = job.text_override.strip()
                else:
                    transcript = self.stt.transcribe(job.wav_bytes)
                session.transcript = transcript
                session.emit_transcript(transcript)
                logger.info("Session %s transcript: %s", job.session_id, transcript)

                if session.cancel.is_set():
                    continue

                intent, confidence, intent_ms = self.intent.classify(transcript)
                session.emit_intent(intent, confidence, intent)
                logger.info(
                    "Session %s intent=%s confidence=%.2f (%.1f ms)",
                    job.session_id,
                    intent,
                    confidence,
                    intent_ms,
                )

                if intent == "ocr":
                    if not job.jpeg_bytes:
                        self._enqueue_tts(job.session_id, self.config.ocr.no_image_message)
                        self._enqueue_tts(job.session_id, EOS_MARKER)
                    else:
                        session.emit_status(PipelineState.READING)
                        lines, ocr_ms = run_apple_vision_ocr_lines(job.jpeg_bytes)
                        logger.info(
                            "Session %s OCR done in %.1f ms (%d lines)",
                            job.session_id,
                            ocr_ms,
                            len(lines),
                        )
                        text = self._select_ocr_text(
                            session,
                            transcript,
                            lines,
                        )
                        normalized = normalize_ocr_text(text)
                        chunks = split_ocr_text(
                            normalized,
                            max_paragraph_chars=self.config.ocr.max_paragraph_chars,
                        )
                        session.response_text = normalized
                        pause_ms = self.config.ocr.paragraph_pause_ms
                        for index, chunk in enumerate(chunks):
                            if session.cancel.is_set():
                                break
                            self._enqueue_tts(
                                job.session_id,
                                chunk,
                                speed=self.config.ocr.audiobook_speed,
                                trailing_pause_ms=pause_ms if index < len(chunks) - 1 else 0,
                            )
                        if not session.cancel.is_set():
                            self._enqueue_tts(job.session_id, EOS_MARKER)
                else:
                    session.emit_status(PipelineState.THINKING)
                    self.llm_queue.put(
                        LLMJob(
                            session_id=job.session_id,
                            transcript=transcript,
                            jpeg_bytes=job.jpeg_bytes,
                        )
                    )
            except Exception as exc:
                logger.exception("STT pipeline error for %s", job.session_id)
                session.emit_error("stt_error", str(exc), recoverable=True)
                self._enqueue_tts(job.session_id, EOS_MARKER)
            finally:
                self.stt_queue.task_done()

    def _llm_worker(self) -> None:
        while True:
            job = self.llm_queue.get()
            if job is None:
                return
            session = self._session_for(job.session_id)
            try:
                if session.cancel.is_set():
                    continue
                parts: list[str] = []
                for sentence in self.llm.stream_sentences(
                    job.transcript,
                    job.jpeg_bytes,
                    session.cancel,
                ):
                    if session.cancel.is_set():
                        break
                    parts.append(sentence)
                    self._enqueue_tts(job.session_id, sentence)
                session.response_text = " ".join(parts).strip()
                if not session.cancel.is_set():
                    self._enqueue_tts(job.session_id, EOS_MARKER)
            except Exception as exc:
                logger.exception("LLM pipeline error for %s", job.session_id)
                session.emit_error("llm_error", str(exc), recoverable=True)
                self._enqueue_tts(job.session_id, EOS_MARKER)
            finally:
                self.llm_queue.task_done()

    def _tts_worker(self) -> None:
        while True:
            item = self.tts_queue.get()
            if item is None:
                return
            session = self._session_for(item.session_id)
            text = item.text
            try:
                if session.cancel.is_set() and text != EOS_MARKER:
                    continue
                is_eos = text == EOS_MARKER
                audio = np.array([], dtype=np.float32)
                sample_rate = self.tts.output_sample_rate
                if not is_eos:
                    audio, sample_rate = self.tts.synthesize(
                        text,
                        speed=item.speed,
                        trailing_pause_ms=item.trailing_pause_ms,
                    )
                self.audio_queue.put(
                    AudioMessage(
                        session_id=item.session_id,
                        audio=audio,
                        text=text if not is_eos else "",
                        is_eos=is_eos,
                        sample_rate=sample_rate,
                    )
                )
            except Exception as exc:
                logger.exception("TTS error for %s", item.session_id)
                session.emit_error("tts_error", str(exc), recoverable=True)
                self.audio_queue.put(
                    AudioMessage(
                        session_id=item.session_id,
                        audio=np.array([], dtype=np.float32),
                        text="",
                        is_eos=True,
                        sample_rate=self.tts.output_sample_rate,
                    )
                )
            finally:
                self.tts_queue.task_done()

    def _sse_worker(self) -> None:
        while True:
            message = self.audio_queue.get()
            if message is None:
                return
            try:
                self._emit_audio_message(message.session_id, message)
            finally:
                self.audio_queue.task_done()

    def _emit_audio_message(self, session_id: str, message: AudioMessage) -> None:
        session = self._session_for(session_id)
        if session.cancel.is_set() and not message.is_eos:
            return

        if len(message.audio) > 0:
            if message.text:
                index = self._text_chunk_counters.get(session_id, 0)
                session.emit_text_chunk(message.text, index)
                session.append_response_text(message.text)
                self._text_chunk_counters[session_id] = index + 1
            wav_bytes = float32_to_wav_bytes(message.audio, message.sample_rate)
            session.emit_audio_chunk(wav_bytes_to_base64(wav_bytes))

        if message.is_eos:
            self._text_chunk_counters.pop(session_id, None)
            session.emit_audio_done()
            self.sessions.release(session)
            self._clear_session_queues(session_id)

    def cancel_session(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        session.cancel.set()
        self._text_chunk_counters.pop(session_id, None)
        self._clear_session_queues(session_id)
        session.emit_audio_done()
        self.sessions.release(session)

    def _clear_session_queues(self, session_id: str) -> None:
        self._purge_queue(self.llm_queue, session_id)
        self._purge_tts_queue(session_id)
        self._purge_audio_queue(session_id)

    def _purge_queue(self, q: queue.Queue, session_id: str) -> None:
        retained = []
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                retained.append(item)
                continue
            if getattr(item, "session_id", None) != session_id:
                retained.append(item)
        for item in retained:
            q.put(item)

    def _purge_tts_queue(self, session_id: str) -> None:
        retained = []
        while True:
            try:
                item = self.tts_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                retained.append(item)
                continue
            if item.session_id != session_id:
                retained.append(item)
        for item in retained:
            self.tts_queue.put(item)

    def _purge_audio_queue(self, session_id: str) -> None:
        _drain_queue(self.audio_queue)
