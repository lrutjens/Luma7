# Luma7 Server

Python intelligence server for Vision Glasses: four-thread pipeline (STT → intent routing → LLM/OCR → TTS → SSE), plus a browser Web UI.

## Requirements

- macOS with Apple Silicon
- Python 3.10+
- `ffmpeg` in PATH
- ~8 GB RAM for FastVLM-1.5B (more for 7B)

## Setup

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
bash scripts/setup_models.sh
python -m luma7 --init-config
```

`setup_models.sh` downloads Kokoro MLX TTS weights, caches HuggingFace models under `models/hub/`, ensures the FastVLM MLX checkpoint is converted locally, and trains the intent classifier.

TTS uses **mlx-audio Kokoro** (`mlx-community/Kokoro-82M-bf16` by default) for native Apple Silicon inference.

After the first setup, restarts load everything from `server/models/` with `HF_HUB_OFFLINE=1` — no HuggingFace requests on startup.

## Run

```bash
source .venv/bin/activate
python -m luma7
```

Open `http://localhost:8080`, paste the auth token from `config.yaml`, and send a text or voice query.

## Configuration

Copy `config.default.yaml` to `config.yaml` or run `--init-config`. Key fields:

| Field | Description |
|-------|-------------|
| `auth_token` | 32-char bearer token for ESP32, web UI, and API |
| `fastvlm.checkpoint` | `llava-fastvithd_1.5b_stage3` or `llava-fastvithd_7b_stage3` |
| `whisper.model` | mlx-whisper HuggingFace repo |
| `kokoro.model` | HuggingFace repo cached under `models/hub/` (default `mlx-community/Kokoro-82M-bf16`) |
| `kokoro.voice` | Kokoro voice id (e.g. `af_sarah`) |

## API

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /api/health` | No | Liveness |
| `POST /query` | Bearer | Binary upload → `session_id` |
| `GET /stream/{session_id}` | Bearer or `?token=` | SSE audio stream |
| `POST /stop/{session_id}` | Bearer | Cancel session |
| `GET /api/status` | Bearer | Model and session state |
| `POST /api/command` | Bearer | Text-only query |

Binary upload format: `uint32_be jpeg_len | jpeg | wav`.
