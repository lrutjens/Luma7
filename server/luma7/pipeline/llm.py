"""FastVLM streaming generation with prompt and vision caches."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from luma7.pipeline.sentence import SentenceBuffer

logger = logging.getLogger(__name__)


class FastVLMEngine:
    def __init__(
        self,
        model_path: Path,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ):
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.model = None
        self.processor = None
        self.config = None
        self.vision_cache = None
        self.prompt_cache_state = None

    def load(self) -> None:
        from mlx_vlm import load
        from mlx_vlm.generate import PromptCacheState
        from mlx_vlm.utils import load_config
        from mlx_vlm.vision_cache import VisionFeatureCache

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"FastVLM model directory not found: {self.model_path}")

        logger.info("Loading FastVLM from %s", self.model_path)
        self.vision_cache = VisionFeatureCache()
        self.prompt_cache_state = PromptCacheState()
        self.config = load_config(str(self.model_path.resolve()))
        self.model, self.processor = load(
            str(self.model_path.resolve()),
            processor_kwargs={"trust_remote_code": True},
        )

    def stream_sentences(
        self,
        transcript: str,
        jpeg_bytes: bytes,
        cancel_event,
    ):
        from mlx_vlm.generate import stream_generate

        if self.model is None or self.processor is None or self.config is None:
            raise RuntimeError("FastVLM not loaded")

        image_path: str | None = None
        if jpeg_bytes:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(jpeg_bytes)
                image_path = tmp.name

        try:
            prompt = self._build_prompt(transcript, image_path is not None)
            image = [image_path] if image_path else None
            gen_kwargs = {
                "max_tokens": int(self.max_tokens),
                "temp": float(self.temperature),
                "vision_cache": self.vision_cache,
                "prompt_cache_state": self.prompt_cache_state,
            }
            if self.top_p < 1.0:
                gen_kwargs["top_p"] = float(self.top_p)

            buffer = SentenceBuffer()
            for chunk in stream_generate(
                self.model,
                self.processor,
                prompt,
                image=image,
                **gen_kwargs,
            ):
                if cancel_event.is_set():
                    return
                for sentence in buffer.feed(chunk.text):
                    yield sentence

            for sentence in buffer.flush():
                yield sentence
        finally:
            if image_path:
                Path(image_path).unlink(missing_ok=True)

    def _build_prompt(self, user_text: str, has_image: bool) -> str:
        from mlx_vlm.prompt_utils import get_chat_template, get_message_json

        assert self.processor is not None and self.config is not None
        messages = []
        if self.system_prompt.strip():
            messages.append(
                get_message_json(
                    self.config["model_type"],
                    self.system_prompt.strip(),
                    role="system",
                    skip_image_token=True,
                    num_images=0,
                )
            )
        messages.append(
            get_message_json(
                self.config["model_type"],
                user_text,
                role="user",
                skip_image_token=not has_image,
                num_images=1 if has_image else 0,
            )
        )
        return get_chat_template(self.processor, messages, add_generation_prompt=True)
