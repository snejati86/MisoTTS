const state = {
  status: null,
  renders: [],
  activeRenderId: null,
  promptFile: null,
  pollTimer: null,
  isRendering: false,
  maxLengthTouched: false,
  logSource: null,
  logs: [],
  logIds: new Set(),
  maxLogs: 200,
  config: null,
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function setBusy(button, busy, label) {
  if (!button) return;
  button.disabled = busy;
  button.classList.toggle("is-busy", busy);
  const span = button.querySelector("span");
  if (span && label) span.textContent = label;
}

function titleCase(value) {
  if (!value) return "Idle";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function setStatusMessage(message, tone = "") {
  els.statusMessage.textContent = message || "";
  els.statusMessage.className = `status-message ${tone}`.trim();
}

function updateStatusUi(status) {
  state.status = status;
  const statusName = titleCase(status.state);
  els.statusLabel.textContent = statusName;
  els.modelState.textContent = statusName;
  els.statusDot.className = `status-dot ${status.state || "idle"}`;
  els.deviceValue.textContent = status.device || "Auto";
  els.dtypeValue.textContent = status.dtype || "Auto";
  const renderCount = status.renders !== undefined && status.renders !== null ? status.renders : state.renders.length || 0;
  els.renderCount.textContent = String(renderCount);
  els.loadValue.textContent = status.load_seconds ? `${status.load_seconds}s` : "--";

  if (state.isRendering) {
    setStatusMessage(status.state === "loading" ? "Warming model for render" : "Rendering audio");
  } else if (status.state === "ready") {
    setStatusMessage("Model ready", "ready");
  } else if (status.state === "loading") {
    setStatusMessage("Warming model");
  } else if (status.state === "error") {
    setStatusMessage(status.error || "Model load failed", "error");
  } else {
    setStatusMessage("Model idle");
  }

  const isLoading = status.state === "loading";
  setBusy(els.warmButton, isLoading, isLoading ? "Warming" : "Warm Model");
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_error) {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

async function refreshStatus() {
  try {
    updateStatusUi(await api("/api/status"));
  } catch (error) {
    setStatusMessage(error.message, "error");
  }
}

async function warmModel() {
  setBusy(els.warmButton, true, "Warming");
  try {
    const payload = {
      device: els.deviceSelect.value,
      dtype: els.dtypeSelect.value,
    };
    updateStatusUi(
      await api("/api/warm", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      }),
    );
  } catch (error) {
    setStatusMessage(error.message, "error");
    setBusy(els.warmButton, false, "Warm Model");
  }
}

async function loadConfig() {
  try {
    state.config = await api("/api/config");
    syncModelControls(state.config);
  } catch (error) {
    setStatusMessage(error.message, "error");
  }
}

function syncModelControls(config) {
  if (!config) return;
  const allowedDevices = config.device_options || ["auto", "cpu"];
  Array.from(els.deviceSelect.options).forEach((option) => {
    option.disabled = allowedDevices.indexOf(option.value) === -1;
  });
  if (allowedDevices.indexOf(els.deviceSelect.value) === -1) {
    els.deviceSelect.value = config.default_device || "auto";
  }
  if (config.default_device === "mps") {
    els.deviceSelect.value = "mps";
  }
  if (config.default_dtype) {
    els.dtypeSelect.value = config.default_dtype;
  }
}

function renderMeta(render) {
  const mode = render.cloned ? "Cloned" : "Native";
  return [
    `${render.duration_seconds}s`,
    `${render.sample_rate} Hz`,
    mode,
    `${render.utterances.length} line${render.utterances.length === 1 ? "" : "s"}`,
  ];
}

function scriptTextOnly(value) {
  return String(value || "")
    .split("\n")
    .map((line) => line.replace(/^\s*(?:\[\d+\]|speaker\s+\d+\s*:)\s*/i, "").trim())
    .filter(Boolean)
    .join(" ");
}

function estimateMaxAudioLengthMs(value) {
  const words = scriptTextOnly(value).split(/\s+/).filter(Boolean).length;
  if (words <= 8) return 1000;
  const seconds = Math.ceil(words / 2.35 + 1.5);
  return Math.min(90000, Math.max(1000, seconds * 1000));
}

function syncMaxLengthToText() {
  if (state.maxLengthTouched) return;
  const estimated = estimateMaxAudioLengthMs(els.textInput.value);
  const current = Number.parseInt(els.maxLengthInput.value || "1000", 10);
  if (estimated > current) {
    els.maxLengthInput.value = String(estimated);
  }
}

function formatLogTime(value) {
  try {
    return new Date(value).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch (_error) {
    return String(value || "").slice(11, 19) || "--:--:--";
  }
}

function logTone(level) {
  const normalized = String(level || "info").toLowerCase();
  if (["critical", "error", "fatal"].indexOf(normalized) !== -1) return "error";
  if (["warning", "warn"].indexOf(normalized) !== -1) return "warning";
  if (normalized === "debug") return "debug";
  return "info";
}

function setLogStreamState(message, tone = "idle") {
  if (!els.logStreamState || !els.logStreamDot) return;
  els.logStreamState.textContent = message;
  els.logStreamDot.className = `stream-dot ${tone}`;
}

function createLogLine(entry) {
  const line = document.createElement("div");
  line.className = `log-line ${logTone(entry.level)}`;

  const time = document.createElement("time");
  time.dateTime = entry.timestamp;
  time.textContent = formatLogTime(entry.timestamp);

  const level = document.createElement("span");
  level.className = "log-level";
  level.textContent = entry.level || "INFO";

  const source = document.createElement("span");
  source.className = "log-source";
  source.textContent = entry.source || "studio";

  const message = document.createElement("span");
  message.className = "log-message";
  message.textContent = entry.message || "";

  line.append(time, level, source, message);
  return line;
}

function renderLogs() {
  if (!els.logList) return;
  els.logList.innerHTML = "";
  if (!state.logs.length) {
    const empty = document.createElement("div");
    empty.className = "log-empty";
    empty.textContent = "Waiting for Studio events";
    els.logList.append(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const entry of state.logs) {
    fragment.append(createLogLine(entry));
  }
  els.logList.append(fragment);
  els.logList.scrollTop = els.logList.scrollHeight;
}

function addLog(entry) {
  if (!entry || state.logIds.has(entry.id)) return;
  state.logIds.add(entry.id);
  state.logs.push(entry);
  while (state.logs.length > state.maxLogs) {
    const removed = state.logs.shift();
    if (removed) state.logIds.delete(removed.id);
  }
}

async function loadLogs() {
  try {
    const entries = await api("/api/logs");
    for (const entry of entries) addLog(entry);
    renderLogs();
  } catch (error) {
    addLog({
      id: `local-${Date.now()}`,
      timestamp: new Date().toISOString(),
      level: "ERROR",
      source: "browser",
      message: error.message,
    });
    renderLogs();
  }
}

function connectLogStream() {
  if (!window.EventSource) {
    setLogStreamState("Polling", "warning");
    window.setInterval(loadLogs, 5000);
    return;
  }

  if (state.logSource) {
    state.logSource.close();
  }

  setLogStreamState("Connecting", "loading");
  state.logSource = new EventSource("/api/logs/stream");
  state.logSource.addEventListener("open", () => {
    setLogStreamState("Live", "ready");
  });
  state.logSource.addEventListener("log", (event) => {
    try {
      addLog(JSON.parse(event.data));
      renderLogs();
    } catch (_error) {
      // Ignore malformed stream events.
    }
  });
  state.logSource.addEventListener("error", () => {
    setLogStreamState("Reconnecting", "warning");
  });
}

function clearVisibleLogs() {
  state.logs = [];
  state.logIds.clear();
  renderLogs();
}

function renderHistory() {
  els.historyList.innerHTML = "";
  if (!state.renders.length) {
    const empty = document.createElement("div");
    empty.className = "history-item";
    empty.innerHTML = "<strong>No renders yet</strong><span>Generated clips appear here</span>";
    els.historyList.append(empty);
    return;
  }

  for (const render of state.renders) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `history-item ${render.render_id === state.activeRenderId ? "active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(render.text.replace(/\s+/g, " ").slice(0, 80))}</strong>
      <span>${renderMeta(render).join(" · ")}</span>
    `;
    button.addEventListener("click", () => selectRender(render));
    els.historyList.append(button);
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const map = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return map[char];
  });
}

async function loadRenders() {
  try {
    state.renders = await api("/api/renders");
    renderHistory();
  } catch (error) {
    setStatusMessage(error.message, "error");
  }
}

async function drawEmptyWaveform() {
  const canvas = els.waveformCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111210";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(100, 214, 194, 0.18)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 72; i += 1) {
    const x = Math.round((i / 71) * width);
    const barHeight = 8 + Math.sin(i * 0.42) * 18 + Math.cos(i * 0.17) * 12;
    ctx.beginPath();
    ctx.moveTo(x, height / 2 - Math.abs(barHeight));
    ctx.lineTo(x, height / 2 + Math.abs(barHeight));
    ctx.stroke();
  }
}

async function drawWaveform(audioUrl) {
  const canvas = els.waveformCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#111210";
  ctx.fillRect(0, 0, width, height);

  try {
    const audioResponse = await fetch(audioUrl);
    const audioBuffer = await audioResponse.arrayBuffer();
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const decoded = await audioContext.decodeAudioData(audioBuffer);
    const samples = decoded.getChannelData(0);
    const columns = Math.min(180, width);
    const stride = Math.max(1, Math.floor(samples.length / columns));

    ctx.lineWidth = Math.max(2, Math.floor(width / columns) - 1);
    for (let i = 0; i < columns; i += 1) {
      let min = 1;
      let max = -1;
      const start = i * stride;
      for (let j = 0; j < stride && start + j < samples.length; j += 1) {
        const sample = samples[start + j];
        if (sample < min) min = sample;
        if (sample > max) max = sample;
      }
      const x = Math.round((i / columns) * width);
      const top = ((1 - max) * height) / 2;
      const bottom = ((1 - min) * height) / 2;
      const gradient = ctx.createLinearGradient(0, top, 0, bottom);
      gradient.addColorStop(0, "#64d6c2");
      gradient.addColorStop(1, "#e7bc63");
      ctx.strokeStyle = gradient;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    audioContext.close();
  } catch (_error) {
    await drawEmptyWaveform();
  }
}

function selectRender(render) {
  state.activeRenderId = render.render_id;
  els.audioPlayer.src = render.audio_url;
  els.downloadLink.href = render.download_url;
  els.downloadLink.classList.remove("disabled");
  els.downloadLink.setAttribute("download", `miso-${render.render_id}.wav`);
  els.renderMeta.innerHTML = renderMeta(render).map((item) => `<span>${escapeHtml(item)}</span>`).join("");
  drawWaveform(render.audio_url);
  renderHistory();
}

function updatePromptFile(file) {
  state.promptFile = file || null;
  els.cloneToggle.checked = Boolean(file);
  if (!file) {
    els.fileName.textContent = "Drop prompt audio";
    els.fileMeta.textContent = "WAV, MP3, M4A";
    return;
  }
  const sizeMb = file.size / 1024 / 1024;
  els.fileName.textContent = file.name;
  els.fileMeta.textContent = `${sizeMb.toFixed(1)} MB`;
}

function buildRenderFormData() {
  syncMaxLengthToText();
  const formData = new FormData();
  formData.set("text", els.textInput.value);
  formData.set("speaker", els.speakerInput.value || "0");
  formData.set("temperature", els.temperatureInput.value);
  formData.set("topk", els.topkInput.value || "50");
  formData.set("max_audio_length_ms", els.maxLengthInput.value || "10000");
  formData.set("device", els.deviceSelect.value || "auto");
  formData.set("dtype", els.dtypeSelect.value || "auto");

  if (els.cloneToggle.checked && state.promptFile) {
    formData.set("prompt_audio", state.promptFile);
    formData.set("prompt_transcript", els.promptTranscriptInput.value);
    formData.set("prompt_speaker", els.promptSpeakerInput.value || "0");
  }
  return formData;
}

async function submitRender(event) {
  event.preventDefault();
  state.isRendering = true;
  setBusy(els.renderButton, true, "Generating");
  setStatusMessage("Rendering");

  try {
    const render = await api("/api/render", {
      method: "POST",
      body: buildRenderFormData(),
    });
    state.renders = [render, ...state.renders.filter((item) => item.render_id !== render.render_id)];
    selectRender(render);
    await refreshStatus();
    state.isRendering = false;
    setStatusMessage("Render ready", "ready");
  } catch (error) {
    state.isRendering = false;
    setStatusMessage(error.message, "error");
  } finally {
    setBusy(els.renderButton, false, "Generate");
  }
}

function bindDragAndDrop() {
  for (const eventName of ["dragenter", "dragover"]) {
    els.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropZone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    els.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("dragging");
    });
  }
  els.dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) updatePromptFile(file);
  });
}

function cacheElements() {
  for (const id of [
    "statusDot",
    "statusLabel",
    "deviceValue",
    "renderCount",
    "warmButton",
    "refreshButton",
    "historyRefreshButton",
    "logsClearButton",
    "renderForm",
    "renderButton",
    "clearButton",
    "textInput",
    "speakerInput",
    "maxLengthInput",
    "topkInput",
    "temperatureInput",
    "temperatureValue",
    "cloneToggle",
    "dropZone",
    "promptAudioInput",
    "fileName",
    "fileMeta",
    "promptTranscriptInput",
    "promptSpeakerInput",
    "modelState",
    "dtypeValue",
    "deviceSelect",
    "dtypeSelect",
    "loadValue",
    "statusMessage",
    "waveformCanvas",
    "audioPlayer",
    "downloadLink",
    "renderMeta",
    "historyList",
    "logList",
    "logStreamDot",
    "logStreamState",
  ]) {
    els[id] = $(id);
  }
}

function bindEvents() {
  els.warmButton.addEventListener("click", warmModel);
  els.refreshButton.addEventListener("click", refreshStatus);
  els.historyRefreshButton.addEventListener("click", loadRenders);
  els.logsClearButton.addEventListener("click", clearVisibleLogs);
  els.renderForm.addEventListener("submit", submitRender);
  els.clearButton.addEventListener("click", () => {
    els.textInput.value = "";
    state.maxLengthTouched = false;
    els.maxLengthInput.value = "1000";
    els.textInput.focus();
  });
  els.textInput.addEventListener("input", syncMaxLengthToText);
  els.maxLengthInput.addEventListener("input", () => {
    state.maxLengthTouched = true;
  });
  els.temperatureInput.addEventListener("input", () => {
    els.temperatureValue.value = els.temperatureInput.value;
  });
  els.promptAudioInput.addEventListener("change", () => {
    updatePromptFile(els.promptAudioInput.files ? els.promptAudioInput.files[0] : null);
  });
  els.cloneToggle.addEventListener("change", () => {
    if (!els.cloneToggle.checked) updatePromptFile(null);
  });
  els.deviceSelect.addEventListener("change", () => {
    if (els.deviceSelect.value === "mps" && ["auto", "float32"].indexOf(els.dtypeSelect.value) !== -1) {
      els.dtypeSelect.value = "float16";
    }
  });
  document.querySelectorAll(".rail-button").forEach((button) => {
    button.addEventListener("click", () => activateRail(button));
  });
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => activateMode(button));
  });
  bindDragAndDrop();
}

function activateRail(button) {
  document.querySelectorAll(".rail-button").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  const label = button.textContent.trim();
  if (label === "Renders") {
    const historyPanel = document.querySelector(".history-panel");
    if (historyPanel) historyPanel.focus({ preventScroll: true });
    els.historyList.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } else if (label === "Logs") {
    const logsPanel = document.querySelector(".logs-panel");
    if (logsPanel) logsPanel.focus({ preventScroll: true });
    els.logList.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } else if (label === "Voices") {
    els.dropZone.focus({ preventScroll: true });
  } else {
    els.textInput.focus({ preventScroll: true });
  }
}

function activateMode(button) {
  document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  const mode = button.textContent.trim();
  if (mode === "Single" && /^\[\d+\]/m.test(els.textInput.value)) {
    const firstLine = els.textInput.value
      .split("\n")
      .map((line) => line.replace(/^\s*\[\d+\]\s*/, "").trim())
      .filter(Boolean)[0];
    if (firstLine) els.textInput.value = firstLine;
  }
  if (mode === "Dialogue" && !/^\s*(?:\[\d+\]|speaker\s+\d+\s*:)/im.test(els.textInput.value)) {
    els.textInput.value = `[${els.speakerInput.value || 0}] ${els.textInput.value.trim()}`;
  }
  els.textInput.focus({ preventScroll: true });
}

async function init() {
  cacheElements();
  bindEvents();
  if (window.lucide) window.lucide.createIcons();
  await drawEmptyWaveform();
  await loadConfig();
  await Promise.all([refreshStatus(), loadRenders(), loadLogs()]);
  connectLogStream();
  state.pollTimer = window.setInterval(refreshStatus, 3500);
}

window.addEventListener("DOMContentLoaded", init);
window.addEventListener("beforeunload", () => {
  if (state.logSource) state.logSource.close();
});
