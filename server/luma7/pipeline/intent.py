"""Intent routing: MiniLM embeddings + logistic regression."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

INTENT_LABELS = ("ocr", "llm")
CLASSIFIER_VERSION = 2

TRAINING_EXAMPLES: list[tuple[str, str]] = [
    ("read this", "ocr"),
    ("what does it say", "ocr"),
    ("what does this say", "ocr"),
    ("what's written there", "ocr"),
    ("read the label", "ocr"),
    ("read that sign", "ocr"),
    ("spell that out", "ocr"),
    ("what does that page say", "ocr"),
    ("read this to me", "ocr"),
    ("what do those words say", "ocr"),
    ("can you read that", "ocr"),
    ("read what's on the screen", "ocr"),
    ("what text is on this", "ocr"),
    ("read the text", "ocr"),
    ("tell me what it says", "ocr"),
    ("what does the sign say", "ocr"),
    ("read the menu", "ocr"),
    ("read the receipt", "ocr"),
    ("what's on the label", "ocr"),
    ("read the ingredients", "ocr"),
    ("what does this document say", "ocr"),
    ("read the caption", "ocr"),
    ("what words are on this", "ocr"),
    ("read that out loud", "ocr"),
    ("what is written here", "ocr"),
    ("read the packaging", "ocr"),
    ("what does the bottle say", "ocr"),
    ("read the poster", "ocr"),
    ("what's printed on this", "ocr"),
    ("read the note", "ocr"),
    ("what does the book say", "ocr"),
    ("what is this", "llm"),
    ("describe this", "llm"),
    ("what am I looking at", "llm"),
    ("what's in front of me", "llm"),
    ("is this safe to eat", "llm"),
    ("what color is that", "llm"),
    ("how do I use this", "llm"),
    ("where am I", "llm"),
    ("what kind of plant is this", "llm"),
    ("what animal is that", "llm"),
    ("is this broken", "llm"),
    ("what brand is this", "llm"),
    ("how many are there", "llm"),
    ("what is happening in this picture", "llm"),
    ("describe the scene", "llm"),
    ("what room is this", "llm"),
    ("is this ripe", "llm"),
    ("what food is this", "llm"),
    ("who is in the photo", "llm"),
    ("what is this object", "llm"),
    ("tell me about this image", "llm"),
    ("what does this look like", "llm"),
    ("is this a dog or a cat", "llm"),
    ("what are the colors in this image", "llm"),
    ("what is the person doing", "llm"),
    ("is this expired", "llm"),
    ("what type of car is this", "llm"),
    ("what building is this", "llm"),
    ("describe what you see", "llm"),
    ("what's in this photo", "llm"),
    ("help me identify this", "llm"),
    ("describe the view", "llm"),
]


class IntentClassifier:
    def __init__(self, encoder_name: str, classifier_path: Path, encoder_path: Path | None = None):
        self.encoder_name = encoder_name
        self.encoder_path = encoder_path
        self.classifier_path = classifier_path
        self.encoder: SentenceTransformer | None = None
        self.classifier: LogisticRegression | None = None

    def load(self) -> None:
        logger.info("Loading intent classifier from %s", self.classifier_path)
        model_path = str(self.encoder_path) if self.encoder_path else self.encoder_name
        local_only = self.encoder_path is not None and self.encoder_path.is_dir()
        self.encoder = SentenceTransformer(model_path, local_files_only=local_only)
        if self.classifier_path.is_file():
            payload = joblib.load(self.classifier_path)
            if self._payload_is_current(payload):
                self.classifier = payload["classifier"]
                return
        logger.info("Training intent classifier")
        self.classifier = self._train()
        self._save(self.classifier)

    def classify(self, transcript: str) -> tuple[str, float, float]:
        text = (transcript or "").strip()
        if not text:
            return "llm", 0.0, 0.0
        assert self.encoder is not None and self.classifier is not None

        t0 = time.perf_counter()
        embedding = self.encoder.encode([text])
        label = self.classifier.predict(embedding)[0]
        confidence = float(self.classifier.predict_proba(embedding)[0].max())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return label, confidence, elapsed_ms

    def _payload_is_current(self, payload) -> bool:
        if not isinstance(payload, dict):
            return False
        if payload.get("version") != CLASSIFIER_VERSION:
            return False
        clf = payload.get("classifier")
        classes = getattr(clf, "classes_", None)
        return classes is not None and set(classes) == set(INTENT_LABELS)

    def _train(self) -> LogisticRegression:
        assert self.encoder is not None
        texts, labels = zip(*TRAINING_EXAMPLES)
        embeddings = self.encoder.encode(list(texts))
        clf = LogisticRegression(max_iter=1000)
        clf.fit(embeddings, list(labels))
        return clf

    def _save(self, classifier: LogisticRegression) -> None:
        self.classifier_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": CLASSIFIER_VERSION, "classifier": classifier}, self.classifier_path)
