# Luma7 — Vision Glasses

Full-stack system for camera-equipped smart glasses with sub-2s spoken responses. See [DESIGN.md](DESIGN.md) for architecture.

## Repository layout

```
Luma7/
├── DESIGN.md           # System design (v4.2)
├── server/             # Python server + Web UI
├── firmware/           # ESP32-S3 ESP-IDF firmware
├── companion/          # iOS BLE relay (planned — Step 3)
└── ml-fastvlm/         # PoC reference only
```

## Build order

1. **Server + Web UI** — `server/README.md`
2. **ESP32 firmware** — `firmware/README.md`
3. **iOS companion** — not yet implemented
4. **Physical glasses** — hardware integration

## Quick start (server)

```bash
cd server
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
bash scripts/setup_models.sh
python -m luma7 --init-config
python -m luma7
```

Open http://localhost:8080 and use the auth token printed in `config.yaml`.
