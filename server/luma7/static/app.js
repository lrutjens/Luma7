const STORAGE_KEY = "luma7-webui";
const SAMPLE_RATE = 22050;

const els = {
  serverUrl: document.getElementById("serverUrl"),
  authToken: document.getElementById("authToken"),
  saveConfig: document.getElementById("saveConfig"),
  serverStatus: document.getElementById("serverStatus"),
  prompt: document.getElementById("prompt"),
  imageFile: document.getElementById("imageFile"),
  sendText: document.getElementById("sendText"),
  recordBtn: document.getElementById("recordBtn"),
  stopBtn: document.getElementById("stopBtn"),
  pipelineState: document.getElementById("pipelineState"),
  transcript: document.getElementById("transcript"),
  intentLabel: document.getElementById("intentLabel"),
  intentConfidence: document.getElementById("intentConfidence"),
  responseText: document.getElementById("responseText"),
  audioQueue: document.getElementById("audioQueue"),
  queueSummary: document.getElementById("queueSummary"),
  errorBox: document.getElementById("errorBox"),
};

let mediaRecorder = null;
let recordedChunks = [];
let audioContext = null;
let playbackTime = 0;
let activeSessionId = null;
let eventSource = null;

const audioQueue = [];
let playbackPumpRunning = false;

function loadSettings() {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  els.serverUrl.value = saved.serverUrl || `${window.location.origin}`;
  els.authToken.value = saved.authToken || "";
}

function saveSettings() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      serverUrl: els.serverUrl.value.trim(),
      authToken: els.authToken.value.trim(),
    }),
  );
}

function authHeaders() {
  return {
    Authorization: `Bearer ${els.authToken.value.trim()}`,
  };
}

function setStatus(text, className = "") {
  els.serverStatus.textContent = text;
  els.serverStatus.className = `status-pill ${className}`.trim();
}

function setPipelineState(state) {
  els.pipelineState.textContent = state;
  els.pipelineState.className = "status-pill busy";
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.classList.remove("hidden");
  setStatus("error", "error");
}

function clearError() {
  els.errorBox.textContent = "";
  els.errorBox.classList.add("hidden");
}

function resetStreamUi() {
  els.transcript.textContent = "—";
  els.intentLabel.textContent = "—";
  els.intentLabel.className = "meta-value";
  els.intentConfidence.textContent = "—";
  els.responseText.textContent = "";
  audioQueue.length = 0;
  renderAudioQueue();
}

function setIntent(intent, confidence, route) {
  const label = (intent || route || "unknown").toUpperCase();
  els.intentLabel.textContent = label;
  els.intentLabel.className = `meta-value intent-${(intent || "").toLowerCase()}`;
  const pct = typeof confidence === "number" ? `${Math.round(confidence * 100)}%` : "—";
  els.intentConfidence.textContent = `Confidence ${pct} · route ${route || intent}`;
}

function appendResponseText(text) {
  const current = els.responseText.textContent;
  els.responseText.textContent = current ? `${current} ${text}` : text;
}

function renderAudioQueue() {
  els.audioQueue.innerHTML = "";
  audioQueue.forEach((item) => {
    const li = document.createElement("li");
    li.className = "queue-item";
    li.dataset.id = item.id;
    li.innerHTML = `
      <span class="queue-index">${item.index + 1}</span>
      <span class="queue-text">${escapeHtml(item.text)}</span>
      <span class="queue-state ${item.status}">${item.status}</span>
    `;
    els.audioQueue.appendChild(li);
  });

  const playing = audioQueue.filter((item) => item.status === "playing").length;
  const queued = audioQueue.filter((item) => item.status === "queued").length;
  const done = audioQueue.filter((item) => item.status === "done").length;
  els.queueSummary.textContent = `${audioQueue.length} chunks · ${playing} playing · ${queued} queued · ${done} done`;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function enqueueTextChunk(text, index) {
  audioQueue.push({
    id: `chunk-${index}`,
    index,
    text,
    status: "queued",
    pcm: null,
    duration: 0,
  });
  renderAudioQueue();
}

function attachAudioToNextQueued(pcmBytes, durationSec) {
  const next = audioQueue.find((item) => item.status === "queued" && !item.pcm);
  if (!next) {
    audioQueue.push({
      id: `chunk-audio-${audioQueue.length}`,
      index: audioQueue.length,
      text: "(audio)",
      status: "queued",
      pcm: pcmBytes,
      duration: durationSec,
    });
    renderAudioQueue();
    pumpPlaybackQueue();
    return;
  }
  next.pcm = pcmBytes;
  next.duration = durationSec;
  renderAudioQueue();
  pumpPlaybackQueue();
}

async function pingServer() {
  try {
    const base = els.serverUrl.value.trim();
    const res = await fetch(`${base}/api/health`);
    if (!res.ok) throw new Error("health check failed");
    setStatus("connected", "live");
  } catch (err) {
    setStatus("disconnected");
  }
}

function ensureAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    playbackTime = audioContext.currentTime;
  }
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }
}

function stripWavHeader(bytes) {
  return bytes.byteLength > 44 ? bytes.slice(44) : bytes;
}

function pcm16ToFloat32(pcm) {
  const floats = new Float32Array(pcm.length / 2);
  const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
  for (let i = 0; i < floats.length; i += 1) {
    floats[i] = view.getInt16(i * 2, true) / 32768;
  }
  return floats;
}

function pcmDurationSec(pcmBytes, sampleRate = SAMPLE_RATE) {
  return pcmBytes.byteLength / 2 / sampleRate;
}

function playPcmBuffer(pcmBytes, sampleRate = SAMPLE_RATE) {
  ensureAudioContext();
  const floats = pcm16ToFloat32(pcmBytes);
  const buffer = audioContext.createBuffer(1, floats.length, sampleRate);
  buffer.copyToChannel(floats, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  const startAt = Math.max(audioContext.currentTime, playbackTime);
  source.start(startAt);
  playbackTime = startAt + buffer.duration;
  return buffer.duration;
}

function pumpPlaybackQueue() {
  if (playbackPumpRunning) return;
  playbackPumpRunning = true;

  const step = () => {
    const ready = audioQueue.find((item) => item.status === "queued" && item.pcm);
    if (!ready) {
      playbackPumpRunning = false;
      return;
    }

    ready.status = "playing";
    renderAudioQueue();
    const duration = playPcmBuffer(ready.pcm, SAMPLE_RATE);
    window.setTimeout(() => {
      ready.status = "done";
      renderAudioQueue();
      step();
    }, Math.max(10, duration * 1000));
  };

  step();
}

function base64ToBytes(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function buildPayload({ wavBytes, jpegBytes }) {
  const jpegLen = jpegBytes?.byteLength || 0;
  const buffer = new ArrayBuffer(4 + jpegLen + wavBytes.byteLength);
  const view = new DataView(buffer);
  view.setUint32(0, jpegLen, false);
  const out = new Uint8Array(buffer);
  if (jpegLen > 0) {
    out.set(jpegBytes, 4);
  }
  out.set(new Uint8Array(wavBytes), 4 + jpegLen);
  return buffer;
}

async function fileToJpegBytes(file) {
  if (!file) return new Uint8Array();
  if (file.type === "image/jpeg") {
    return new Uint8Array(await file.arrayBuffer());
  }
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  return new Uint8Array(await blob.arrayBuffer());
}

function closeStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  els.stopBtn.disabled = true;
}

function openStream(sessionId) {
  closeStream();
  activeSessionId = sessionId;
  els.stopBtn.disabled = false;
  playbackTime = audioContext ? audioContext.currentTime : 0;
  resetStreamUi();

  const base = els.serverUrl.value.trim();
  const url = new URL(`${base}/stream/${sessionId}`);
  url.searchParams.set("token", els.authToken.value.trim());

  eventSource = new EventSource(url);

  eventSource.addEventListener("status", (event) => {
    const payload = JSON.parse(event.data);
    setPipelineState(payload.state);
  });

  eventSource.addEventListener("transcript", (event) => {
    const payload = JSON.parse(event.data);
    els.transcript.textContent = payload.text || "—";
  });

  eventSource.addEventListener("intent", (event) => {
    const payload = JSON.parse(event.data);
    setIntent(payload.intent, payload.confidence, payload.route);
  });

  eventSource.addEventListener("text_chunk", (event) => {
    const payload = JSON.parse(event.data);
    if (payload.text) {
      appendResponseText(payload.text);
      enqueueTextChunk(payload.text, payload.index ?? audioQueue.length);
    }
  });

  eventSource.addEventListener("audio_chunk", (event) => {
    const wavBytes = base64ToBytes(event.data);
    const pcm = stripWavHeader(wavBytes);
    const duration = pcmDurationSec(pcm, SAMPLE_RATE);
    attachAudioToNextQueued(pcm, duration);
    setPipelineState("speaking");
  });

  eventSource.addEventListener("error", (event) => {
    try {
      const payload = JSON.parse(event.data);
      showError(payload.message || "Pipeline error");
    } catch (_) {
      showError("Stream error");
    }
    closeStream();
  });

  eventSource.addEventListener("audio_done", () => {
    setPipelineState("done");
    closeStream();
  });

  eventSource.onerror = () => {
    if (eventSource.readyState === EventSource.CLOSED) {
      closeStream();
    }
  };
}

async function postQuery(payloadBuffer) {
  clearError();
  setPipelineState("uploading");

  const base = els.serverUrl.value.trim();
  const res = await fetch(`${base}/query`, {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/octet-stream",
    },
    body: payloadBuffer,
  });

  if (res.status === 503) {
    showError("Server is busy with another session.");
    return;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    showError(err.detail || err.error || `Request failed (${res.status})`);
    return;
  }

  const { session_id: sessionId } = await res.json();
  openStream(sessionId);
}

async function sendTextQuery() {
  const text = els.prompt.value.trim();
  if (!text) {
    showError("Enter a prompt.");
    return;
  }

  const base = els.serverUrl.value.trim();
  const res = await fetch(`${base}/api/command`, {
    method: "POST",
    headers: {
      ...authHeaders(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      respond_in_body: false,
      image_base64: await imageAsBase64(),
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    showError(err.detail || err.error || `Command failed (${res.status})`);
    return;
  }

  const { session_id: sessionId } = await res.json();
  openStream(sessionId);
}

async function imageAsBase64() {
  const file = els.imageFile.files[0];
  if (!file) return null;
  const bytes = await fileToJpegBytes(file);
  let binary = "";
  bytes.forEach((b) => {
    binary += String.fromCharCode(b);
  });
  return btoa(binary);
}

async function sendBinaryQuery(wavBlob) {
  const wavBytes = await wavBlob.arrayBuffer();
  const jpegBytes = await fileToJpegBytes(els.imageFile.files[0]);
  const payload = buildPayload({ wavBytes, jpegBytes });
  await postQuery(payload);
}

async function stopActiveSession() {
  if (!activeSessionId) return;
  const base = els.serverUrl.value.trim();
  await fetch(`${base}/stop/${activeSessionId}`, {
    method: "POST",
    headers: authHeaders(),
  });
  closeStream();
  setPipelineState("stopped");
}

async function startRecording() {
  clearError();
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  recordedChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) recordedChunks.push(event.data);
  };
  mediaRecorder.onstop = async () => {
    stream.getTracks().forEach((track) => track.stop());
    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
    const wavBlob = await webmToWav(blob);
    await sendBinaryQuery(wavBlob);
  };
  mediaRecorder.start();
  els.recordBtn.textContent = "Stop recording";
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    els.recordBtn.textContent = "Record voice";
  }
}

async function webmToWav(blob) {
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  const channel = audioBuffer.getChannelData(0);
  const pcm = new Int16Array(channel.length);
  for (let i = 0; i < channel.length; i += 1) {
    pcm[i] = Math.max(-32768, Math.min(32767, Math.round(channel[i] * 32767)));
  }

  const dataSize = pcm.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i += 1) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true);
  view.setUint32(28, 16000 * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);
  new Uint8Array(buffer, 44).set(new Uint8Array(pcm.buffer));
  await audioCtx.close();
  return new Blob([buffer], { type: "audio/wav" });
}

els.saveConfig.addEventListener("click", () => {
  saveSettings();
  pingServer();
});

els.sendText.addEventListener("click", sendTextQuery);
els.stopBtn.addEventListener("click", stopActiveSession);
els.recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
    return;
  }
  try {
    await startRecording();
  } catch (err) {
    showError("Microphone permission required.");
  }
});

loadSettings();
pingServer();
