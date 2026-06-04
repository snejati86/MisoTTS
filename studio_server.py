from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import io
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional
from uuid import uuid4

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torchaudio
from fastapi import Body, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from generator import DEFAULT_MISO_TTS_REPO_ID, Segment, TokenizerAccessError, load_miso_8b

APP_ROOT = Path(__file__).resolve().parent
WEB_ROOT = APP_ROOT / "web"
OUTPUT_ROOT = Path(os.environ.get("MISO_TTS_STUDIO_OUTPUT_DIR", APP_ROOT / "outputs" / "studio")).resolve()
RENDER_ROOT = OUTPUT_ROOT / "renders"
LOG_ROOT = OUTPUT_ROOT / "logs"
LOG_FILE = LOG_ROOT / "studio.log"
MAX_PROMPT_AUDIO_MB = int(os.environ.get("MISO_TTS_MAX_PROMPT_AUDIO_MB", "100"))
LOG_BUFFER_LIMIT = int(os.environ.get("MISO_TTS_LOG_BUFFER_LIMIT", "500"))


@dataclass(frozen=True)
class ModelConfig:
    device: str
    dtype_name: str
    model_source: str


@dataclass
class ParsedUtterance:
    text: str
    speaker: int


class WarmRequest(BaseModel):
    force: bool = False
    device: Optional[str] = None
    dtype: Optional[str] = None
    model_source: Optional[str] = None


class RenderSummary(BaseModel):
    render_id: str
    text: str
    speaker: int
    duration_seconds: float
    sample_rate: int
    created_at: str
    audio_url: str
    download_url: str
    cloned: bool
    utterances: list[dict[str, Any]]


class LogEntry(BaseModel):
    id: int
    timestamp: str
    level: str
    source: str
    message: str


class StatusResponse(BaseModel):
    state: str
    device: Optional[str] = None
    dtype: Optional[str] = None
    model_source: Optional[str] = None
    loaded_at: Optional[str] = None
    load_started_at: Optional[str] = None
    load_seconds: Optional[float] = None
    error: Optional[str] = None
    renders: int = 0


class DiagnosticsResponse(BaseModel):
    state: str
    configured_device: Optional[str] = None
    configured_dtype: Optional[str] = None
    loaded: bool
    generator_device: Optional[str] = None
    model_parameter_device: Optional[str] = None
    model_parameter_dtype: Optional[str] = None
    sample_rate: Optional[int] = None
    mps_available: bool
    mps_current_allocated_bytes: Optional[int] = None
    mps_driver_allocated_bytes: Optional[int] = None
    mps_recommended_max_bytes: Optional[int] = None
    mps_fallback_enabled: bool
    cuda_available: bool
    cuda_allocated_bytes: Optional[int] = None


class ConfigResponse(BaseModel):
    max_prompt_audio_mb: int
    output_dir: str
    default_device: str
    default_dtype: str
    mps_available: bool
    device_options: list[str]
    dtype_options: list[str]
    script_examples: list[str]


class LogBroker:
    def __init__(self, limit: int) -> None:
        self._entries: deque[LogEntry] = deque(maxlen=limit)
        self._subscribers: set[tuple[asyncio.AbstractEventLoop, asyncio.Queue[LogEntry]]] = set()
        self._lock = threading.Lock()
        self._next_id = 0

    def publish(self, level: str, source: str, message: str) -> None:
        clean_message = str(message).strip()
        if len(clean_message) > 4_000:
            clean_message = f"{clean_message[:3997]}..."
        with self._lock:
            self._next_id += 1
            entry = LogEntry(
                id=self._next_id,
                timestamp=_utc_now(),
                level=level.upper(),
                source=source,
                message=clean_message,
            )
            self._entries.append(entry)
            subscribers = list(self._subscribers)

        for loop, queue in subscribers:
            loop.call_soon_threadsafe(self._enqueue, queue, entry)

    def publish_record(self, record: logging.LogRecord) -> None:
        self.publish(record.levelname, record.name, record.getMessage())

    def snapshot(self, after: int = 0) -> list[LogEntry]:
        with self._lock:
            return [entry for entry in self._entries if entry.id > after]

    def subscribe(self) -> tuple[asyncio.Queue[LogEntry], tuple[asyncio.AbstractEventLoop, asyncio.Queue[LogEntry]]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=200)
        subscriber = (loop, queue)
        with self._lock:
            self._subscribers.add(subscriber)
        return queue, subscriber

    def unsubscribe(self, subscriber: tuple[asyncio.AbstractEventLoop, asyncio.Queue[LogEntry]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    @staticmethod
    def _enqueue(queue: asyncio.Queue[LogEntry], entry: LogEntry) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(entry)


class StudioLogHandler(logging.Handler):
    def __init__(self, broker: LogBroker) -> None:
        super().__init__()
        self._broker = broker
        self._miso_studio_stream = True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._broker.publish_record(record)
        except Exception:
            pass


log_stream = LogBroker(LOG_BUFFER_LIMIT)
logger = logging.getLogger("misotts.studio")


def _configure_logging() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    logging.captureWarnings(True)

    stream_handler = StudioLogHandler(log_stream)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler._miso_studio_file = True  # type: ignore[attr-defined]
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

    for name in ("misotts.studio", "uvicorn.error", "py.warnings"):
        target_logger = logging.getLogger(name)
        target_logger.setLevel(logging.INFO)
        if not any(getattr(handler, "_miso_studio_stream", False) for handler in target_logger.handlers):
            target_logger.addHandler(stream_handler)
        if name == "misotts.studio" and not any(getattr(handler, "_miso_studio_file", False) for handler in target_logger.handlers):
            target_logger.addHandler(file_handler)

    logger.info("Studio log stream ready; writing file logs to %s", LOG_FILE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse_log_entry(entry: LogEntry) -> str:
    return f"id: {entry.id}\nevent: log\ndata: {entry.model_dump_json()}\n\n"


def _sse_ping() -> str:
    return "event: ping\ndata: {}\n\n"


def _select_device(requested: Optional[str]) -> str:
    if requested and requested != "auto":
        if requested == "mps" and not _mps_available():
            raise ValueError("MPS is not available in this PyTorch environment.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _select_dtype_name(device: str, requested: Optional[str]) -> str:
    if requested and requested != "auto":
        return requested
    if device == "cuda":
        return "bfloat16"
    if device == "mps":
        return "float16"
    return "float32"


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def _torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _default_model_config() -> ModelConfig:
    device = _select_device(os.environ.get("MISO_TTS_DEVICE", "auto"))
    dtype_name = _select_dtype_name(device, os.environ.get("MISO_TTS_DTYPE", "auto"))
    return ModelConfig(
        device=device,
        dtype_name=dtype_name,
        model_source=os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID),
    )


def _request_config(request: WarmRequest) -> ModelConfig:
    base = _default_model_config()
    device = _select_device(request.device or base.device)
    requested_dtype = request.dtype
    if requested_dtype is None and request.device:
        requested_dtype = "auto"
    dtype_name = _select_dtype_name(device, requested_dtype or base.dtype_name)
    return ModelConfig(
        device=device,
        dtype_name=dtype_name,
        model_source=request.model_source or base.model_source,
    )


def _load_generator(config: ModelConfig):
    logger.info("Loading MisoTTS model from %s on %s with %s", config.model_source, config.device, config.dtype_name)
    return load_miso_8b(
        device=config.device,
        model_path_or_repo_id=config.model_source,
        dtype=_torch_dtype(config.dtype_name),
    )


def _release_accelerator_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _mps_available():
        torch.mps.empty_cache()


class ModelManager:
    def __init__(self) -> None:
        self._generator = None
        self._config: Optional[ModelConfig] = None
        self._state = "idle"
        self._error: Optional[str] = None
        self._loaded_at: Optional[str] = None
        self._load_started_at: Optional[str] = None
        self._load_seconds: Optional[float] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._state_lock = asyncio.Lock()
        self.generation_lock = asyncio.Lock()

    async def begin_warm(self, request: Optional[WarmRequest] = None) -> StatusResponse:
        request = request or WarmRequest()
        config = _request_config(request)
        async with self._state_lock:
            task_running = self._task is not None and not self._task.done()
            same_config = self._config == config
            if task_running:
                logger.info("Warm request ignored because model loading is already running")
                return self.status()
            if self._state == "ready" and same_config and not request.force:
                logger.info("Warm request skipped because the model is already ready on %s with %s", config.device, config.dtype_name)
                return self.status()

            self._state = "loading"
            self._error = None
            self._load_started_at = _utc_now()
            self._load_seconds = None
            self._config = config
            self._generator = None
            _release_accelerator_cache()
            logger.info("Warmup started on %s with %s", config.device, config.dtype_name)
            self._task = asyncio.create_task(self._warm_background(config))
            return self.status()

    async def _warm_background(self, config: ModelConfig) -> None:
        started = time.perf_counter()
        try:
            generator = await asyncio.to_thread(_load_generator, config)
        except TokenizerAccessError as exc:
            await self._mark_error(str(exc), started)
        except Exception as exc:
            await self._mark_error(f"{type(exc).__name__}: {exc}", started)
        else:
            async with self._state_lock:
                self._generator = generator
                self._state = "ready"
                self._error = None
                self._loaded_at = _utc_now()
                self._load_seconds = round(time.perf_counter() - started, 2)
                logger.info("Warmup complete in %.2fs", self._load_seconds)

    async def _mark_error(self, message: str, started: float) -> None:
        async with self._state_lock:
            self._state = "error"
            self._error = message
            self._load_seconds = round(time.perf_counter() - started, 2)
            self._generator = None
            logger.error("Warmup failed after %.2fs: %s", self._load_seconds, message)

    async def ensure_ready(self, request: Optional[WarmRequest] = None) -> Any:
        request = request or WarmRequest()
        target_config = _request_config(request)
        if self._state != "ready" or self._generator is None or self._config != target_config:
            if self._state == "loading" and self._config != target_config:
                logger.warning(
                    "Requested %s/%s while %s/%s is still loading; waiting for current load first",
                    target_config.device,
                    target_config.dtype_name,
                    self._config.device if self._config else "unknown",
                    self._config.dtype_name if self._config else "unknown",
                )
            else:
                logger.info("Model is not ready for %s/%s; starting warmup before render", target_config.device, target_config.dtype_name)
            await self.begin_warm(request)
        if self._task is not None and not self._task.done():
            await self._task
        if self._state == "ready" and self._generator is not None and self._config != target_config:
            logger.info("Switching loaded model from %s/%s to %s/%s", self._config.device, self._config.dtype_name, target_config.device, target_config.dtype_name)
            await self.begin_warm(
                WarmRequest(
                    force=True,
                    device=target_config.device,
                    dtype=target_config.dtype_name,
                    model_source=target_config.model_source,
                )
            )
            if self._task is not None and not self._task.done():
                await self._task
        if self._state != "ready" or self._generator is None:
            detail = self._error or "MisoTTS model is not ready yet."
            logger.error("Model is unavailable: %s", detail)
            raise HTTPException(status_code=503, detail=detail)
        if self._config != target_config:
            detail = f"Loaded model config {self._config} did not match requested config {target_config}."
            logger.error(detail)
            raise HTTPException(status_code=503, detail=detail)
        return self._generator

    def status(self) -> StatusResponse:
        return StatusResponse(
            state=self._state,
            device=self._config.device if self._config else None,
            dtype=self._config.dtype_name if self._config else None,
            model_source=self._config.model_source if self._config else None,
            loaded_at=self._loaded_at,
            load_started_at=self._load_started_at,
            load_seconds=self._load_seconds,
            error=self._error,
            renders=len(list(RENDER_ROOT.glob("*.json"))) if RENDER_ROOT.exists() else 0,
        )

    def diagnostics(self) -> DiagnosticsResponse:
        model_parameter_device = None
        model_parameter_dtype = None
        sample_rate = None
        generator_device = None
        if self._generator is not None:
            generator_device = str(getattr(self._generator, "device", None))
            sample_rate = int(getattr(self._generator, "sample_rate", 0)) or None
            try:
                parameter = next(self._generator._model.parameters())
            except Exception:
                parameter = None
            if parameter is not None:
                model_parameter_device = str(parameter.device)
                model_parameter_dtype = str(parameter.dtype)

        mps_current = None
        mps_driver = None
        mps_recommended = None
        if _mps_available():
            for attr_name, setter in (
                ("current_allocated_memory", "current"),
                ("driver_allocated_memory", "driver"),
                ("recommended_max_memory", "recommended"),
            ):
                attr = getattr(torch.mps, attr_name, None)
                if attr is None:
                    continue
                try:
                    value = int(attr())
                except Exception:
                    continue
                if setter == "current":
                    mps_current = value
                elif setter == "driver":
                    mps_driver = value
                else:
                    mps_recommended = value

        return DiagnosticsResponse(
            state=self._state,
            configured_device=self._config.device if self._config else None,
            configured_dtype=self._config.dtype_name if self._config else None,
            loaded=self._generator is not None,
            generator_device=generator_device,
            model_parameter_device=model_parameter_device,
            model_parameter_dtype=model_parameter_dtype,
            sample_rate=sample_rate,
            mps_available=_mps_available(),
            mps_current_allocated_bytes=mps_current,
            mps_driver_allocated_bytes=mps_driver,
            mps_recommended_max_bytes=mps_recommended,
            mps_fallback_enabled=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
            cuda_available=torch.cuda.is_available(),
            cuda_allocated_bytes=int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,
        )


manager = ModelManager()
_configure_logging()


def _parse_script(text: str, default_speaker: int) -> list[ParsedUtterance]:
    stripped = text.strip()
    if not stripped:
        raise HTTPException(status_code=400, detail="Text is required.")

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    parsed: list[ParsedUtterance] = []
    labelled_count = 0
    pattern = re.compile(r"^(?:\[?speaker\s*)?(\d+)\]?\s*[:|-]\s*(.+)$", re.IGNORECASE)
    bracket_pattern = re.compile(r"^\[(\d+)\]\s*(.+)$")

    for line in lines:
        match = pattern.match(line) or bracket_pattern.match(line)
        if match:
            labelled_count += 1
            parsed.append(ParsedUtterance(text=match.group(2).strip(), speaker=int(match.group(1))))
        else:
            parsed.append(ParsedUtterance(text=line, speaker=default_speaker))

    if labelled_count == 0:
        return [ParsedUtterance(text=stripped, speaker=default_speaker)]
    return parsed


def _validate_generation_settings(text: str, temperature: float, topk: int, max_audio_length_ms: int) -> None:
    if len(text.strip()) > 8_000:
        raise HTTPException(status_code=400, detail="Text is too long. Keep a render under 8,000 characters.")
    if not 0.1 <= temperature <= 2.0:
        raise HTTPException(status_code=400, detail="Temperature must be between 0.1 and 2.0.")
    if not 1 <= topk <= 512:
        raise HTTPException(status_code=400, detail="Top-k must be between 1 and 512.")
    if not 1_000 <= max_audio_length_ms <= 90_000:
        raise HTTPException(status_code=400, detail="Max audio length must be between 1,000 and 90,000 ms.")


def _text_without_speaker_labels(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\[\d+\]|speaker\s+\d+\s*:)\s*", "", line, flags=re.IGNORECASE).strip()
        if cleaned:
            lines.append(cleaned)
    return " ".join(lines)


def _estimate_audio_length_ms(text: str) -> int:
    word_count = len(re.findall(r"\S+", _text_without_speaker_labels(text)))
    if word_count <= 8:
        return 1_000
    seconds = int((word_count / 2.35) + 2)
    return min(90_000, max(1_000, seconds * 1_000))


def _log_if_likely_truncated(text: str, max_audio_length_ms: int) -> None:
    estimated_ms = _estimate_audio_length_ms(text)
    if max_audio_length_ms < estimated_ms:
        logger.warning(
            "Render max_audio_length_ms=%s may truncate this text; estimated speech budget is about %s ms",
            max_audio_length_ms,
            estimated_ms,
        )


def _decode_prompt_audio(audio_bytes: bytes, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(io.BytesIO(audio_bytes))
    if waveform.ndim != 2 or waveform.size(0) == 0:
        raise ValueError("Prompt audio did not contain a valid waveform.")
    mono = waveform.mean(dim=0)
    if source_rate != sample_rate:
        mono = torchaudio.functional.resample(mono, orig_freq=source_rate, new_freq=sample_rate)
    return mono


def _generate_sync(
    generator: Any,
    text: str,
    speaker: int,
    temperature: float,
    topk: int,
    max_audio_length_ms: int,
    prompt_audio_bytes: Optional[bytes],
    prompt_transcript: Optional[str],
    prompt_speaker: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], bool]:
    context: list[Segment] = []
    cloned = False
    if prompt_audio_bytes:
        if not prompt_transcript or not prompt_transcript.strip():
            raise ValueError("Prompt transcript is required when prompt audio is provided.")
        prompt_audio = _decode_prompt_audio(prompt_audio_bytes, generator.sample_rate)
        context.append(Segment(speaker=prompt_speaker, text=prompt_transcript.strip(), audio=prompt_audio))
        cloned = True

    rendered_segments: list[Segment] = []
    utterance_meta: list[dict[str, Any]] = []
    parsed_utterances = _parse_script(text, speaker)
    logger.info("Generation worker started for %s utterance(s)", len(parsed_utterances))
    for index, utterance in enumerate(parsed_utterances, start=1):
        history = context + rendered_segments
        logger.info("Generating utterance %s/%s for speaker %s", index, len(parsed_utterances), utterance.speaker)
        audio = generator.generate(
            text=utterance.text,
            speaker=utterance.speaker,
            context=history,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )
        rendered_segments.append(Segment(speaker=utterance.speaker, text=utterance.text, audio=audio))
        utterance_meta.append(
            {
                "speaker": utterance.speaker,
                "text": utterance.text,
                "duration_seconds": round(float(audio.numel()) / float(generator.sample_rate), 3),
            }
        )

    if not rendered_segments:
        raise ValueError("No text was provided for generation.")
    logger.info("Generation worker complete")
    return torch.cat([segment.audio for segment in rendered_segments], dim=0), utterance_meta, cloned


def _metadata_path(render_id: str) -> Path:
    return RENDER_ROOT / f"{render_id}.json"


def _audio_path(render_id: str) -> Path:
    return RENDER_ROOT / f"{render_id}.wav"


def _read_render(render_id: str) -> RenderSummary:
    metadata_file = _metadata_path(render_id)
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Render not found.")
    return RenderSummary(**json.loads(metadata_file.read_text()))


def _write_render(
    audio: torch.Tensor,
    sample_rate: int,
    text: str,
    speaker: int,
    cloned: bool,
    utterances: list[dict[str, Any]],
) -> RenderSummary:
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    render_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    audio_file = _audio_path(render_id)
    torchaudio.save(str(audio_file), audio.unsqueeze(0).cpu(), sample_rate)
    summary = RenderSummary(
        render_id=render_id,
        text=text,
        speaker=speaker,
        duration_seconds=round(float(audio.numel()) / float(sample_rate), 3),
        sample_rate=sample_rate,
        created_at=_utc_now(),
        audio_url=f"/api/renders/{render_id}/audio",
        download_url=f"/api/renders/{render_id}/download",
        cloned=cloned,
        utterances=utterances,
    )
    _metadata_path(render_id).write_text(summary.model_dump_json(indent=2))
    logger.info(
        "Render %s saved: %.3fs at %s Hz, cloned=%s",
        render_id,
        summary.duration_seconds,
        sample_rate,
        cloned,
    )
    return summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("Studio startup complete; renders directory is %s", RENDER_ROOT)
    if os.environ.get("MISO_TTS_AUTOLOAD", "1") != "0":
        logger.info("Autoload is enabled; starting model warmup")
        await manager.begin_warm()
    else:
        logger.info("Autoload is disabled; model will warm on demand")
    yield
    logger.info("Studio shutdown")


app = FastAPI(
    title="MisoTTS Studio API",
    description="Warm-loaded local API for MisoTTS generation, playback, downloads, and voice cloning.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return manager.status()


@app.get("/api/logs", response_model=list[LogEntry])
async def logs(after: int = 0) -> list[LogEntry]:
    return log_stream.snapshot(after)


@app.get("/api/logs/stream")
async def log_events(after: int = 0, last_event_id: Optional[str] = Header(None, alias="Last-Event-ID")) -> StreamingResponse:
    replay_after = after
    if last_event_id and last_event_id.isdigit():
        replay_after = max(replay_after, int(last_event_id))

    async def event_stream():
        for entry in log_stream.snapshot(replay_after):
            yield _sse_log_entry(entry)

        queue, subscriber = log_stream.subscribe()
        try:
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield _sse_ping()
                    continue
                yield _sse_log_entry(entry)
        finally:
            log_stream.unsubscribe(subscriber)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics() -> DiagnosticsResponse:
    return manager.diagnostics()


@app.get("/api/config", response_model=ConfigResponse)
async def config() -> ConfigResponse:
    default_config = _default_model_config()
    return ConfigResponse(
        max_prompt_audio_mb=MAX_PROMPT_AUDIO_MB,
        output_dir=str(OUTPUT_ROOT),
        default_device=default_config.device,
        default_dtype=default_config.dtype_name,
        mps_available=_mps_available(),
        device_options=["auto", "cpu", "mps"] if _mps_available() else ["auto", "cpu"],
        dtype_options=["auto", "float32", "float16", "bfloat16"],
        script_examples=[
            "[0] I'm testing a warm-loaded local render.",
            "[1] This line uses a second speaker in the same render.",
            "Speaker 0: Dialogue lines can use speaker labels too.",
        ],
    )


@app.post("/api/warm", response_model=StatusResponse)
async def warm(request: WarmRequest = Body(default_factory=WarmRequest)) -> StatusResponse:
    try:
        return await manager.begin_warm(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/render", response_model=RenderSummary)
async def render(
    text: str = Form(...),
    speaker: int = Form(0),
    temperature: float = Form(0.9),
    topk: int = Form(50),
    max_audio_length_ms: int = Form(10_000),
    device: str = Form("auto"),
    dtype: str = Form("auto"),
    prompt_transcript: str = Form(""),
    prompt_speaker: int = Form(0),
    prompt_audio: Optional[UploadFile] = File(None),
) -> RenderSummary:
    _validate_generation_settings(text, temperature, topk, max_audio_length_ms)
    _log_if_likely_truncated(text, max_audio_length_ms)
    try:
        generator = await manager.ensure_ready(WarmRequest(device=device, dtype=dtype))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "Render requested: chars=%s speaker=%s temperature=%.2f topk=%s max_audio_length_ms=%s device=%s dtype=%s clone=%s",
        len(text.strip()),
        speaker,
        temperature,
        topk,
        max_audio_length_ms,
        device,
        dtype,
        prompt_audio is not None,
    )

    prompt_audio_bytes: Optional[bytes] = None
    if prompt_audio is not None:
        prompt_audio_bytes = await prompt_audio.read()
        if len(prompt_audio_bytes) > MAX_PROMPT_AUDIO_MB * 1024 * 1024:
            logger.warning("Prompt audio rejected because it exceeds %s MB", MAX_PROMPT_AUDIO_MB)
            raise HTTPException(status_code=413, detail=f"Prompt audio must be under {MAX_PROMPT_AUDIO_MB} MB.")
        if not prompt_audio_bytes:
            prompt_audio_bytes = None
        else:
            logger.info("Prompt audio received: %.2f MB", len(prompt_audio_bytes) / 1024 / 1024)

    async with manager.generation_lock:
        try:
            started = time.perf_counter()
            audio, utterances, cloned = await asyncio.to_thread(
                _generate_sync,
                generator,
                text,
                speaker,
                temperature,
                topk,
                max_audio_length_ms,
                prompt_audio_bytes,
                prompt_transcript,
                prompt_speaker,
            )
            logger.info("Render generation completed in %.2fs", time.perf_counter() - started)
        except ValueError as exc:
            logger.warning("Render rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Render failed")
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return _write_render(audio, generator.sample_rate, text, speaker, cloned, utterances)


@app.get("/api/renders", response_model=list[RenderSummary])
async def renders() -> list[RenderSummary]:
    if not RENDER_ROOT.exists():
        return []
    summaries = []
    for metadata_file in sorted(RENDER_ROOT.glob("*.json"), reverse=True):
        try:
            summaries.append(RenderSummary(**json.loads(metadata_file.read_text())))
        except Exception:
            continue
    return summaries


@app.get("/api/renders/{render_id}", response_model=RenderSummary)
async def render_details(render_id: str) -> RenderSummary:
    return _read_render(render_id)


@app.get("/api/renders/{render_id}/audio")
async def render_audio(render_id: str) -> FileResponse:
    _read_render(render_id)
    audio_file = _audio_path(render_id)
    if not audio_file.exists():
        logger.warning("Audio playback requested for missing render file %s", render_id)
        raise HTTPException(status_code=404, detail="Render audio file not found.")
    logger.info("Serving playback audio for render %s", render_id)
    return FileResponse(audio_file, media_type="audio/wav")


@app.get("/api/renders/{render_id}/download")
async def render_download(render_id: str) -> FileResponse:
    _read_render(render_id)
    audio_file = _audio_path(render_id)
    if not audio_file.exists():
        logger.warning("Download requested for missing render file %s", render_id)
        raise HTTPException(status_code=404, detail="Render audio file not found.")
    logger.info("Serving download for render %s", render_id)
    return FileResponse(audio_file, media_type="audio/wav", filename=f"miso-{render_id}.wav")


@app.get("/", response_model=None)
async def index():
    index_file = WEB_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {
            "name": "MisoTTS Studio",
            "status_url": "/api/status",
            "warm_url": "/api/warm",
            "render_url": "/api/render",
            "message": "Backend is ready. Select a UI direction to build the Studio frontend.",
        }
    )


if WEB_ROOT.exists():
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="assets")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("studio_server:app", host="127.0.0.1", port=7860, reload=False)
