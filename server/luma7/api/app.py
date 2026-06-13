"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from luma7.auth import require_auth
from luma7.config import ServerConfig, load_config, save_config
from luma7.logging import get_ring_handler, setup_logging
from luma7.pipeline.engine import PipelineEngine
from luma7.pipeline.intent import IntentClassifier
from luma7.pipeline.llm import FastVLMEngine
from luma7.pipeline.session import SessionManager
from luma7.pipeline.stt import SpeechToText
from luma7.pipeline.tts import TextToSpeech
from luma7.pipeline.types import STTJob
from luma7.models.hub_cache import configure_hub_environment
from luma7.protocol import parse_query_body

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class CommandRequest(BaseModel):
    text: str
    respond_in_body: bool = False
    image_base64: str | None = None


class ConfigPatch(BaseModel):
    host: str | None = None
    port: int | None = None
    fastvlm_checkpoint: str | None = Field(default=None, alias="fastvlm.checkpoint")
    fastvlm_system_prompt: str | None = Field(default=None, alias="fastvlm.system_prompt")


def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    setup_logging(cfg.logging.level, cfg.logging.ring_size)
    configure_hub_environment(cfg.models_root)

    app = FastAPI(title="Luma7 Vision Glasses Server")
    sessions = SessionManager()
    auth = require_auth(cfg.auth_token)

    stt = SpeechToText(str(cfg.whisper_model_path))
    intent = IntentClassifier(
        cfg.intent.encoder,
        cfg.intent_classifier_path,
        encoder_path=cfg.intent_encoder_path,
    )
    llm = FastVLMEngine(
        model_path=cfg.fastvlm_mlx_path,
        system_prompt=cfg.fastvlm.system_prompt,
        temperature=cfg.fastvlm.temperature,
        top_p=cfg.fastvlm.top_p,
        max_tokens=cfg.fastvlm.max_tokens,
    )
    tts = TextToSpeech(
        model_path=str(cfg.kokoro_mlx_path),
        voice=cfg.kokoro.voice,
        speed=cfg.kokoro.speed,
        lang=cfg.kokoro.lang,
        output_sample_rate=cfg.kokoro.output_sample_rate,
    )
    engine = PipelineEngine(cfg, sessions, stt, intent, llm, tts)

    app.state.config = cfg
    app.state.sessions = sessions
    app.state.engine = engine

    @app.on_event("startup")
    def _startup() -> None:
        def _load():
            try:
                engine.load_models()
                engine.start()
            except Exception:
                logger.exception("Model loading failed")

        threading.Thread(target=_load, name="model-loader", daemon=True).start()

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/status", dependencies=[Depends(auth)])
    async def api_status():
        return {
            "models_ready": engine.models_ready,
            "active_session": sessions.active_session_id,
            "busy": sessions.is_busy(),
            "fastvlm_checkpoint": cfg.fastvlm.checkpoint,
        }

    @app.get("/api/config", dependencies=[Depends(auth)])
    async def api_get_config():
        return {
            "host": cfg.host,
            "port": cfg.port,
            "models_dir": cfg.models_dir,
            "fastvlm": {
                "checkpoint": cfg.fastvlm.checkpoint,
                "system_prompt": cfg.fastvlm.system_prompt,
                "temperature": cfg.fastvlm.temperature,
                "max_tokens": cfg.fastvlm.max_tokens,
            },
            "whisper": {"model": cfg.whisper.model},
            "kokoro": {
                "voice": cfg.kokoro.voice,
                "output_sample_rate": cfg.kokoro.output_sample_rate,
            },
        }

    @app.patch("/api/config", dependencies=[Depends(auth)])
    async def api_patch_config(body: dict):
        if "fastvlm" in body and isinstance(body["fastvlm"], dict):
            for key, value in body["fastvlm"].items():
                if hasattr(cfg.fastvlm, key):
                    setattr(cfg.fastvlm, key, value)
        for key in ("host", "port", "models_dir"):
            if key in body:
                setattr(cfg, key, body[key])
        save_config(cfg)
        return await api_get_config()

    @app.get("/api/logs", dependencies=[Depends(auth)])
    async def api_logs(n: int = 100, level: str | None = None):
        entries = get_ring_handler().recent(n=n, level=level)
        return {
            "logs": [
                {
                    "timestamp": item.timestamp,
                    "level": item.level,
                    "message": item.message,
                }
                for item in entries
            ]
        }

    @app.post("/query", dependencies=[Depends(auth)])
    async def post_query(request: Request):
        if sessions.is_busy():
            return JSONResponse(status_code=503, content={"error": "busy"})
        body = await request.body()
        try:
            payload = parse_query_body(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session = sessions.create_session()
        engine.submit(
            STTJob(
                session_id=session.session_id,
                wav_bytes=payload.wav,
                jpeg_bytes=payload.jpeg,
            )
        )
        return {"session_id": session.session_id}

    @app.get("/stream/{session_id}", dependencies=[Depends(auth)])
    async def stream_session(session_id: str, request: Request):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")

        async def event_generator():
            while True:
                if await request.is_disconnected():
                    engine.cancel_session(session_id)
                    break
                batch = session.drain_sse()
                for event_type, data in batch:
                    yield f"event: {event_type}\ndata: {data}\n\n"
                    if event_type == "audio_done":
                        return
                if session._closed:
                    return
                await asyncio.sleep(0.05)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @app.post("/stop/{session_id}", dependencies=[Depends(auth)])
    async def stop_session(session_id: str):
        engine.cancel_session(session_id)
        return {}

    @app.post("/api/command", dependencies=[Depends(auth)])
    async def api_command(body: CommandRequest):
        if sessions.is_busy():
            return JSONResponse(status_code=503, content={"error": "busy"})
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")

        import base64
        import io
        import wave

        jpeg_bytes = b""
        if body.image_base64:
            jpeg_bytes = base64.b64decode(body.image_base64)

        silent_wav = _silent_wav_bytes()
        session = sessions.create_session()
        engine.submit(
            STTJob(
                session_id=session.session_id,
                wav_bytes=silent_wav,
                jpeg_bytes=jpeg_bytes,
                text_override=text,
            )
        )

        if not body.respond_in_body:
            return {"session_id": session.session_id}

        deadline = asyncio.get_event_loop().time() + 120
        audio_chunks: list[str] = []
        response_text = ""
        while asyncio.get_event_loop().time() < deadline:
            for event_type, data in session.drain_sse():
                if event_type == "audio_chunk":
                    audio_chunks.append(data)
                elif event_type == "status":
                    pass
                elif event_type == "error":
                    payload = json.loads(data)
                    raise HTTPException(status_code=500, detail=payload.get("message", "error"))
                elif event_type == "audio_done":
                    response_text = session.response_text
                    return {
                        "session_id": session.session_id,
                        "text": response_text,
                        "audio_base64": "".join(audio_chunks),
                    }
            if session._closed:
                break
            await asyncio.sleep(0.05)

        raise HTTPException(status_code=504, detail="command timed out")

    @app.get("/")
    async def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        raise HTTPException(status_code=404, detail="web ui missing")

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def _silent_wav_bytes(duration_s: float = 0.25, sample_rate: int = 16000) -> bytes:
    import struct

    num_samples = int(duration_s * sample_rate)
    data_size = num_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + (b"\x00\x00" * num_samples)
