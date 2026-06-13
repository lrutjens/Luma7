#!/usr/bin/env python3
"""Train and save the intent classifier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from luma7.config import load_config
from luma7.models.hub_cache import configure_hub_environment, ensure_hub_repo
from luma7.pipeline.intent import IntentClassifier


def main() -> None:
    config = load_config()
    configure_hub_environment(config.models_root)
    encoder_path = ensure_hub_repo(config.models_root, config.intent.encoder)
    classifier = IntentClassifier(
        config.intent.encoder,
        config.intent_classifier_path,
        encoder_path=encoder_path,
    )
    classifier.load()
    print(f"Intent classifier ready at {config.intent_classifier_path}")


if __name__ == "__main__":
    main()
