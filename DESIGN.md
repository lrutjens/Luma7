# Vision Glasses — System Design
### Version 4.2 — Intent Routing with Embedding Classifier

---

## What This Borrows From GLaDOS

GLaDOS achieves sub-600ms round-trip response time through one core insight: the pipeline is not sequential. While the LLM generates sentence two, TTS synthesizes sentence one, and the speaker plays the audio from sentence zero — all simultaneously, in separate threads, connected by queues. You hear the first word of the response before the model has finished thinking.

This project applies the same architecture to a remote client (ESP32 glasses). Everything on the server side is structurally identical to GLaDOS. The transport layer uses HTTP POST for uploads and Server-Sent Events (SSE) for streaming audio back — chosen over WebSockets for stability and straightforward implementation on ESP32.

---

## The Latency Goal

**Time from end of speech to first audio output is the only metric that matters.**

**LLM path (visual Q&A, description, general queries):**
```
Touch released (end of speech)
  → POST payload to server             ~100–400ms (WiFi) / ~1–3s (BLE)
  → STT transcribes                    ~150–300ms
  → Intent classifier                  ~5ms
  → LLM generates first sentence       ~500ms–1.5s 
  → TTS synthesizes first sentence     ~100–200ms  (parallel with LLM sentence 2)
  → first SSE audio_chunk event arrives glasses
  → first word heard
Total (WiFi path):                     ~1–2.5 seconds
Total (BLE path):                      ~2.5–5 seconds
```

**OCR path (reading text from image):**
```
Touch released (end of speech)
  → POST payload to server             ~100–400ms (WiFi) / ~1–3s (BLE)
  → STT transcribes                    ~150–300ms
  → Intent classifier                  ~5ms
  → Apple Vision OCR                   ~50–150ms
  → TTS synthesizes first chunk        ~100–200ms
  → first word heard
Total (WiFi path):                     ~400ms–1 second
Total (BLE path):                      ~1.5–3 seconds
```

The OCR path skips FastVLM entirely, making it significantly faster than the LLM path and removing TTFT variance from the equation. The FastVLM prompt cache is critical for the LLM path. After the first query the system prompt KV state is cached. Every subsequent query only prefills the new image and transcript tokens — directly reducing TTFT on every turn after warmup.

---

## Components

### 1. Python Server
The intelligence of the system. Runs on Mac mini. Four threads connected by three queues — mirroring the GLaDOS engine, with an intent routing branch in the STT thread that splits the pipeline between OCR and LLM paths. Exposes HTTP endpoints for upload and SSE streaming, a REST API for the management app, and serves the web UI as static files.

### 2. ESP32 Firmware
Runs on Seeed XIAO ESP32-S3 Sense. Capacitive touch trigger, I2S mic recording with VAD end detection, PWDN-gated camera capture only when needed, PSRAM ring buffer, I2S audio playback, HTTP client for POST and SSE.

### 3. iOS Companion App (Swift/SwiftUI)
BLE relay when away from home. Also provides server config UI and manual text controls.

### 4. Web UI
Single page served by the Python server. Uses `EventSource` for SSE natively in the browser. Voice or text input, optional image, real-time text and audio response.

---

## Server Pipeline Architecture

Four threads, three queues. The STT thread now doubles as the intent router — after transcription, a synchronous classifier call (~5ms) decides which branch the pipeline takes. The LLM thread is bypassed entirely on OCR queries.

```
[HTTP POST /query received]
    ↓ returns session_id immediately
    ↓ binary payload handed to pipeline
    ↓
┌──────────────────────────────────┐
│   STT Thread + Intent Router     │  mlx-whisper: audio bytes → transcript
│                                  │  all-MiniLM-L6-v2: embed transcript
│                                  │  LogisticRegression: → "ocr" or "llm"
└────────────┬─────────────────────┘
             ↓
     ┌────────────────┐
     │                │
  [llm route]    [ocr route]
     │                │
     ↓                ↓
┌──────────┐    ┌─────────────────┐
│   LLM    │    │  Apple Vision   │
│  Thread  │    │  OCR (inline,   │
│ FastVLM  │    │  STT thread)    │
│ streaming│    │  image → text   │
│ sentence │    │  split into     │
│ buffering│    │  TTS-sized      │
│ → EOS    │    │  chunks + EOS   │
└────┬─────┘    └────────┬────────┘
     │                   │
     └─────────┬─────────┘
               ↓ sentence string → tts_queue
┌─────────────────┐
│   TTS Thread    │  Kokoro ONNX: sentence → float32 audio array
│                 │  Wraps in AudioMessage(audio, text, is_eos)
│                 │  Pushes → audio_queue
└────────┬────────┘
         ↓ AudioMessage → audio_queue
┌─────────────────┐
│  SSE Emitter    │  Reads AudioMessage from audio_queue
│  Thread         │  Converts float32 array → PCM 16-bit → WAV bytes
│                 │  Base64-encodes WAV bytes
│                 │  Writes SSE event to session's response stream
│                 │  On is_eos: writes audio_done event, closes stream
└─────────────────┘
```

On the OCR path, Apple Vision runs synchronously in the STT thread immediately after classification. The resulting text is split into natural chunks (sentence boundaries, or ~6-word windows if no punctuation is present), wrapped as sentence strings with a terminal `<EOS>`, and pushed directly onto `tts_queue`. The LLM thread never wakes. The TTS and SSE stages are identical for both paths.

If an OCR intent is classified but the payload contains no image (JPEG length = 0), the pipeline pushes a single short string — "There's no image to read." — directly to `tts_queue` and terminates.

While TTS is synthesizing sentence one, the LLM is generating sentence two (LLM path). On the OCR path all chunks are available immediately, so TTS starts the moment the first chunk is pushed, with subsequent chunks already queued. The SSE emitter behavior is identical in both cases.

### Intent Routing

The classifier is loaded at server startup and kept in memory. It adds ~5ms to the STT thread's work on every query — well within noise for the latency budget.

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (~22MB). Encodes the transcript into a 384-dimensional embedding. A `LogisticRegression` classifier trained on ~60 labeled examples maps the embedding to `ocr` or `llm`.

**Training data shape:**

| Label | Example prompts |
|-------|----------------|
| `ocr` | "read this", "what does it say", "what's written there", "read the label", "read that sign", "spell that out", "what does that page say", "read this to me", "what do those words say" |
| `llm` | "what is this", "describe this", "what am I looking at", "what's in front of me", "is this safe to eat", "what color is that", "how do I use this", "where am I", "what kind of plant is this" |

**Implementation:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# At server startup
encoder = SentenceTransformer("all-MiniLM-L6-v2")
classifier = joblib.load("intent_classifier.joblib")  # pre-trained

def classify_intent(transcript: str) -> str:
    embedding = encoder.encode([transcript])          # shape (1, 384)
    return classifier.predict(embedding)[0]           # "ocr" or "llm"
```

**Training script** (run once, produces `intent_classifier.joblib`):

```python
examples = [
    ("read this", "ocr"),
    ("what does it say", "ocr"),
    # ... ~60 total
]
texts, labels = zip(*examples)
embeddings = encoder.encode(list(texts))
clf = LogisticRegression(max_iter=1000).fit(embeddings, labels)
joblib.dump(clf, "intent_classifier.joblib")
```

The classifier is deliberately narrow — only two classes, ~60 examples, linear model. It does not need to understand nuance, only separate "read the text pixels" from everything else. Retraining takes seconds if new failure cases are observed.

### Thinking Tag Handling

The LLM thread monitors the token stream. When `<|channel>thought` is detected, forwarding to the TTS queue is suspended. When `<channel|>` closes the block, forwarding resumes. Thinking content is discarded. Only the final answer reaches TTS and the user's ears. Thinking stays enabled because disabling it degrades model quality without saving compute — the model reasons badly and the reasoning leaks into the visible output.

### Sentence Boundary Detection

The LLM thread accumulates tokens and splits on terminal punctuation (`.` `!` `?`) followed by whitespace, respecting decimals and abbreviations. Minimum chunk is ~4 words to avoid TTS receiving fragments too short to sound natural. The system prompt is written to bias the model toward short, naturally-spoken phrases, which creates frequent boundaries without requiring a special delimiter.

### Interruption

When a new touch fires during playback, the ESP32 POSTs to `/stop/{session_id}`. The server sets a cancellation event on the active pipeline task. The LLM thread stops generating. The TTS queue and audio queue are drained. The SSE stream is closed. The pipeline resets. The ESP32 independently flushes its ring buffer and stops I2S playback on touch.

---

## HTTP / SSE Protocol

### Phase 1 — Upload

```
POST /query
Content-Type: application/octet-stream
Authorization: Bearer <token>

Body:
  Bytes 0–3:    uint32 big-endian — JPEG length N (0 if no image)
  Bytes 4–4+N:  JPEG image data
  Bytes 4+N–:   WAV audio data (PCM 16-bit mono 16kHz, 44-byte header)

Response 200:
  {"session_id": "abc123"}

Response 503:
  {"error": "busy"} — prior session still active
```

The server returns the session_id as fast as possible after receiving the full body. The pipeline starts processing immediately. The response is not blocked on the pipeline.

### Phase 2 — SSE Stream

```
GET /stream/{session_id}
Authorization: Bearer <token>
Accept: text/event-stream

Response headers:
  Content-Type: text/event-stream
  Cache-Control: no-cache
  Connection: keep-alive
```

The server holds this connection open and pushes events as the pipeline produces them. The ESP32 opens this connection immediately after receiving the session_id from Phase 1. The pipeline takes long enough (STT + LLM TTFT) that the SSE connection is always established before the first audio chunk is ready.

### SSE Event Types

All events follow standard SSE format: `event:` line, `data:` line, blank line separator.

**status event** — pipeline state updates
```
event: status
data: {"state":"transcribing"}

event: status
data: {"state":"thinking"}

event: status
data: {"state":"reading"}

event: status
data: {"state":"speaking"}

```

`thinking` is emitted on the LLM path when FastVLM begins generating. `reading` is emitted on the OCR path when Apple Vision begins processing the image.

**audio_chunk event** — base64-encoded WAV audio
```
event: audio_chunk
data: <base64-encoded WAV bytes (PCM 16-bit mono 22050Hz, 44-byte header)>

```

**audio_done event** — response complete, stream will close
```
event: audio_done
data: {}

```

**error event** — pipeline failure
```
event: error
data: {"code":"stt_error","message":"transcription failed","recoverable":true}

```

### Phase 3 — Interruption (if needed)

```
POST /stop/{session_id}
Authorization: Bearer <token>

Response 200: {}
```

### Management REST Endpoints

```
GET  /api/health          unauthenticated liveness check
GET  /api/status          server state, model loaded, active session
GET  /api/config          current non-sensitive config
PATCH /api/config         update config at runtime
POST /api/command         text-only query (manual control from companion app)
GET  /api/logs            recent log lines (?n=100&level=info)
```

`POST /api/command` accepts `{"text": "...", "respond_in_body": true}`. When `respond_in_body` is true, the response includes the full base64 WAV and response text synchronously — used by the companion app's manual controls UI so it doesn't need to open an SSE stream.

### Authentication

All endpoints require `Authorization: Bearer <token>` except `/api/health`. Token is a 32-character alphanumeric string configured once, stored in server config, and on the ESP32 in NVS. Constant-time comparison on the server side.

---

## ESP32 SSE Implementation

The ESP32 HTTP client (`esp_http_client`) handles SSE as a streaming GET with `HTTP_METHOD_GET` and a registered `on_data` event handler. The handler maintains a small line-buffer state machine:

- Accumulate bytes until `\n`
- If line starts with `event:` → store the event type
- If line starts with `data:` → store the data payload
- If line is empty (blank line = event boundary) → dispatch the accumulated event+data pair
- After dispatch, reset event type and data buffer

On `audio_chunk` dispatch: base64-decode the data field → strip the 44-byte WAV header from the decoded bytes → write raw PCM to PSRAM ring buffer. The I2S task on Core 1 reads the ring buffer continuously and feeds MAX98357A.

On `audio_done` dispatch: set a flag. When the ring buffer drains completely, return to idle state.

The ESP32 runs Phase 1 (POST) and Phase 2 (GET/SSE) sequentially — POST completes first (returns session_id), then immediately opens the SSE GET. No parallelism needed on the ESP32 side since the pipeline takes long enough that the SSE stream is always ready before data arrives.

---

## BLE Relay with SSE

The companion app relays both phases:

**Phase 1 relay:** ESP32 sends binary payload over BLE to companion app in chunks → companion app reassembles → POSTs to server → receives session_id → sends session_id back over BLE.

**Phase 2 relay:** Companion app opens SSE stream to server using the session_id. As `audio_chunk` events arrive, companion app base64-decodes them, rechunks the raw WAV bytes into BLE MTU-sized frames, and notifies the ESP32 over BLE. ESP32 reassembles frames, strips WAV header, pushes PCM to ring buffer. For other event types (status, audio_done, error), companion app sends them as small JSON control messages over BLE.

**Stop relay:** ESP32 writes a stop control message over BLE → companion app POSTs to `/stop/{session_id}`.

This separation of upload (POST) and streaming (SSE) makes the BLE relay cleaner than a WebSocket relay would have been — the two phases are independently managed connections with clear start/end points.

---

## Web UI SSE

The browser uses the native `EventSource` API:

```
EventSource(url)  →  automatic reconnect, native SSE parsing
```

The web UI POSTs the payload first (via `fetch`), gets the session_id, then opens an `EventSource` to `/stream/{session_id}`. It listens for `audio_chunk` events, base64-decodes them, strips the WAV header, and feeds the PCM bytes into a Web Audio API buffer queue for real-time playback. Text tokens arrive via `status` events and stream into a display area. No WebSocket needed — `EventSource` plus a single `fetch` POST covers everything.

---

## Touch Trigger

No wake word. The XIAO ESP32-S3's GPIO pins have hardware capacitive touch sensing via the ESP-IDF touch sensor peripheral. A copper pad or small metal contact on the glasses arm is wired to a touch-capable GPIO. The touch driver handles debouncing and threshold configuration in NVS.

**Touch start:** recording begins, camera briefly powers on (PWDN low), AF triggers, JPEG captured, camera powers off (PWDN high). The whole camera cycle is under one second and runs while the mic is recording.

**Touch end or VAD silence:** recording stops, payload is built, POST begins.

**Touch during playback:** POST to `/stop/{session_id}`, flush ring buffer, stop I2S playback, return to idle.


---

## Thermal Management

The device should minimize sustained heat in the glasses frame and battery enclosure.

- Keep the camera fully powered down except during an active capture window.
- Do not leave the camera running across the full speech session if one JPEG is enough for the query.
- Prefer short, bounded bursts of camera, CPU, and radio activity over long continuous runs.
- If the device starts warming up during repeated use, reduce optional work first: shorten capture windows, lower camera frequency, and avoid unnecessary retries.

## Hardware Summary

- **MCU:** Seeed XIAO ESP32-S3 Sense
- **Camera:** OV5640 (5MP autofocus) — off by default via PWDN, on only during capture
- **Mic:** INMP441 I2S MEMS
- **Speaker:** 1W 8Ω via MAX98357A I2S amp (not currently, attached)
- **Touch:** Copper pad on arm → touch-capable GPIO, ESP-IDF touch peripheral
- **Battery:** 200–300mAh LiPo, USB-C via XIAO SGM40567
- **Core 0:** touch, mic, camera, HTTP client, BLE
- **Core 1:** I2S ring buffer playback only

---

## Build Order

**Step 1 — Server + Web UI**
Python server with all four threads. Web UI using `EventSource`. Validate the full pipeline via browser: POST payload → SSE stream → audio plays. Proves the architecture before any ESP32 work.

**Step 2 — ESP32 breadboard**
Wire XIAO + INMP441 + MAX98357A + speaker + touch pad. Implement HTTP POST and SSE client. Test WiFi direct path end to end.

**Step 3 — iOS Companion App + BLE relay**
Add BLE relay for the away-from-home path.

**Step 4 — Physical glasses**
Mount in frames, route cables, install battery, tune camera and touch placement.
