# Luma7 ESP32 Firmware

ESP-IDF firmware for Seeed XIAO ESP32-S3 Sense. Touch-to-talk, I2S mic, gated camera capture, HTTP POST + SSE playback over WiFi.

## Wiring (breadboard)

| Signal | GPIO | Notes |
|--------|------|-------|
| INMP441 BCLK | 8 | I2S mic |
| INMP441 WS | 7 | |
| INMP441 SD | 9 | |
| MAX98357A BCLK | 2 | I2S speaker |
| MAX98357A LRC | 3 | |
| MAX98357A DIN | 4 | |
| Touch pad | 1 | Capacitive pad on arm |

Adjust pins in `main/app_audio.c` and `main/app_touch.c` if your harness differs.

## NVS configuration

Flash once, then set WiFi and server credentials (e.g. via `idf.py monitor` and a future provisioning tool). Defaults are in `app_config.c`:

- `server_url`: `http://192.168.1.100:8080`
- `auth_token`: must match server `config.yaml`

Keys: `wifi_ssid`, `wifi_pass`, `server_url`, `auth_token`, `touch_thr`.

## Build

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/tty.usbmodem* flash monitor
```

## Runtime flow

1. **Touch down** — start mic recording, capture one JPEG, power camera down.
2. **Touch up or VAD silence** — build payload, `POST /query`.
3. **SSE** — `GET /stream/{session_id}`, decode `audio_chunk` events into PSRAM ring buffer, Core 1 plays I2S.
4. **Touch during playback** — `POST /stop/{session_id}`, flush playback.
