"""Entry point: python -m luma7"""

from __future__ import annotations

import argparse

import uvicorn

from luma7.api.app import create_app
from luma7.config import load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Luma7 Vision Glasses server")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--init-config", action="store_true", help="Write config.yaml from defaults")
    args = parser.parse_args()

    config_path = None
    if args.config:
        from pathlib import Path

        config_path = Path(args.config)

    config = load_config(config_path)
    if args.init_config:
        save_config(config)
        print(f"Wrote config to {config.config_path}")
        print(f"Auth token: {config.auth_token}")

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
