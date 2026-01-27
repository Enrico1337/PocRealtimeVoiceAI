"""
Orchestrator Main: FastAPI server with Pipecat voice pipeline.

Supports three transport modes:
- Daily Mode (default): Uses Daily.co for hosted WebRTC (works with vast.ai)
- Local Mode: Uses SmallWebRTC for direct connections (for LAN/localhost)
- Twilio Mode: Uses Twilio Media Streams for telephone calls
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.stt_service import SegmentedSTTService
from typing import AsyncGenerator
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler,
    SmallWebRTCRequest,
)
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TextFrame, Frame, EndFrame, TTSAudioRawFrame, TranscriptionFrame, ErrorFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from .pipeline import RAGProcessor, InterruptionHandler, ResponseLogger
from .rag import RAGService
from .settings import Settings, get_settings, TransportMode
from .telemetry import SessionMetrics, setup_logging
from .text_utils import sanitize_for_tts
from .transport import TransportFactory, TransportConfig, DailyRoomInfo, TwilioCallInfo
from .transport.local import get_local_ice_servers, get_local_ice_servers_for_client
from .events import event_manager

logger = logging.getLogger(__name__)

# Global instances
rag_service: Optional[RAGService] = None
settings: Optional[Settings] = None
transport_factory: Optional[TransportFactory] = None
webrtc_handler: Optional[SmallWebRTCRequestHandler] = None

# Path to client HTML files
CLIENTS_DIR = Path(__file__).parent / "clients"


class FasterWhisperSTTService(SegmentedSTTService):
    """Custom STT Service for faster-whisper-server.

    Uses SegmentedSTTService to leverage VAD events for speech detection,
    preventing hallucinations on silence.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        language: str = "de",
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language
        self._client = None
        logger.info(f"FasterWhisperSTTService initialized: model={model}, language={language}")

    async def _get_client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None:
            import aiohttp
            self._client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0)
            )
        return self._client

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio segment.

        Note: SegmentedSTTService calls this only when user stops speaking,
        with properly formatted audio data.
        """
        if not audio or len(audio) < 1000:  # Skip very short/empty audio
            return

        try:
            client = await self._get_client()

            import aiohttp
            form = aiohttp.FormData()
            form.add_field('file', audio, filename='audio.wav', content_type='audio/wav')
            form.add_field('model', self._model)
            form.add_field('language', self._language)
            form.add_field('response_format', 'json')

            logger.debug(f"STT request: language={self._language}, audio_size={len(audio)}")

            async with client.post(
                f"{self._base_url}/v1/audio/transcriptions",
                data=form
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"STT API error: {response.status} - {error_text}")
                    yield ErrorFrame(error=f"STT API error: {response.status}")
                    return

                result = await response.json()
                text = result.get("text", "").strip()

                if text:
                    logger.info(f"STT transcription: '{text}'")
                    yield TranscriptionFrame(text=text, user_id="", timestamp="")

        except Exception as e:
            logger.error(f"STT error: {e}")
            yield ErrorFrame(error=f"STT error: {e}")

    async def cleanup(self):
        """Cleanup HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None


class HTTPTTSService(TTSService):
    """TTS Service that calls our custom HTTP TTS endpoint.

    Uses lazy initialization for the HTTP client to avoid issues when
    the service is not directly in the pipeline (e.g., used by SentenceAggregator).
    """

    def __init__(self, base_url: str, sample_rate: int = 24000):
        super().__init__(sample_rate=sample_rate)
        self.base_url = base_url.rstrip("/")
        self._client = None
        # Store sample_rate directly since we're not in the pipeline
        # and start() won't be called to initialize _sample_rate
        self._output_sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        """Override to return the stored sample rate."""
        return self._output_sample_rate

    async def _get_client(self):
        """Lazy initialization of HTTP client."""
        if self._client is None:
            import aiohttp
            self._client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60.0)
            )
        return self._client

    async def cleanup(self):
        """Cleanup HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def run_tts(self, text: str):
        """Generate speech from text and yield audio frames."""
        if not text.strip():
            return

        try:
            client = await self._get_client()
            async with client.post(
                f"{self.base_url}/v1/audio/speech",
                json={
                    "model": "chatterbox",
                    "input": text,
                    "voice": "default",
                    "response_format": "pcm",
                    "speed": 1.0
                }
            ) as response:
                response.raise_for_status()
                audio_data = await response.read()

                # Yield audio in chunks (100ms at 24kHz, 16-bit mono)
                chunk_size = 4800
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    yield chunk

        except Exception as e:
            logger.error(f"TTS error: {e}")


class SentenceAggregator(FrameProcessor):
    """Aggregates text into sentences for TTS.

    Collects text frames and generates TTS audio when sentence boundaries are detected.
    Includes error handling to prevent pipeline breakage on TTS failures.

    Features:
    - Text sanitization to prevent TTS gibberish
    - Fallback chunking when buffer exceeds max length without sentence ending
    - Comma-based chunking for more natural speech pauses
    """

    def __init__(self, tts_service: HTTPTTSService):
        super().__init__()
        self.tts_service = tts_service
        self._buffer = ""
        self._sentence_endings = ".!?"
        # Fallback settings to prevent long pauses
        self._max_buffer_length = 150  # Fallback after 150 chars without sentence ending
        self._chunk_on_comma = True    # Allow chunking at commas for natural pauses
        self._min_chunk_length = 20    # Minimum chunk length before comma chunking

    async def _generate_tts_audio(self, text: str, direction: FrameDirection):
        """Generate TTS audio for text and push frames.

        Args:
            text: Text to synthesize
            direction: Frame direction for pushing frames
        """
        if not text or not text.strip():
            return

        # Sanitize text for TTS to prevent gibberish
        sanitized_text = sanitize_for_tts(text)

        if not sanitized_text:
            logger.debug(f"TTS skipped: text sanitized to empty string")
            return

        logger.debug(f"TTS input (raw): '{text[:100]}{'...' if len(text) > 100 else ''}'")
        logger.debug(f"TTS input (sanitized): '{sanitized_text[:100]}{'...' if len(sanitized_text) > 100 else ''}'")

        try:
            async for audio_chunk in self.tts_service.run_tts(sanitized_text):
                audio_frame = TTSAudioRawFrame(
                    audio=audio_chunk,
                    sample_rate=self.tts_service.sample_rate,
                    num_channels=1
                )
                await self.push_frame(audio_frame, direction)
        except Exception as e:
            logger.error(f"SentenceAggregator TTS error for '{sanitized_text[:50]}...': {e}")
            # Don't re-raise - allow pipeline to continue without audio

    def _find_chunk_boundary(self) -> int:
        """Find the best position to chunk the buffer.

        Returns:
            Position to split at (exclusive), or -1 if no suitable boundary found
        """
        # First, try to find a sentence ending
        end_pos = -1
        for char in self._sentence_endings:
            pos = self._buffer.find(char)
            if pos != -1 and (end_pos == -1 or pos < end_pos):
                end_pos = pos

        if end_pos != -1:
            return end_pos + 1  # Include the sentence ending

        # If buffer exceeds max length, find fallback boundary
        if len(self._buffer) >= self._max_buffer_length:
            # Try comma first (for natural pauses in German)
            if self._chunk_on_comma:
                # Find last comma within the buffer
                comma_pos = self._buffer.rfind(',', self._min_chunk_length, self._max_buffer_length)
                if comma_pos != -1:
                    logger.debug(f"Buffer overflow ({len(self._buffer)} chars), chunking at comma")
                    return comma_pos + 1  # Include the comma

            # Try semicolon or colon
            for char in ';:':
                pos = self._buffer.rfind(char, self._min_chunk_length, self._max_buffer_length)
                if pos != -1:
                    logger.debug(f"Buffer overflow ({len(self._buffer)} chars), chunking at '{char}'")
                    return pos + 1

            # Last resort: chunk at last space before max length
            space_pos = self._buffer.rfind(' ', self._min_chunk_length, self._max_buffer_length)
            if space_pos != -1:
                logger.debug(f"Buffer overflow ({len(self._buffer)} chars), chunking at space")
                return space_pos + 1

            # Ultimate fallback: force chunk at max length
            logger.warning(f"Buffer overflow ({len(self._buffer)} chars), forcing chunk at max length")
            return self._max_buffer_length

        return -1  # No suitable boundary found yet

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and frame.text:
            self._buffer += frame.text

            # Process chunks while we have suitable boundaries
            while self._buffer:
                chunk_pos = self._find_chunk_boundary()

                if chunk_pos > 0:
                    # Extract chunk
                    chunk = self._buffer[:chunk_pos].strip()
                    self._buffer = self._buffer[chunk_pos:].lstrip()

                    if chunk:
                        # Generate TTS for this chunk
                        await self._generate_tts_audio(chunk, direction)
                else:
                    # No suitable boundary found, wait for more text
                    break

        elif isinstance(frame, EndFrame):
            # Flush remaining buffer
            if self._buffer.strip():
                await self._generate_tts_audio(self._buffer.strip(), direction)
                self._buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def cleanup(self):
        """Cleanup TTS service resources."""
        if self.tts_service:
            await self.tts_service.cleanup()


def create_pipeline_components(session_id: Optional[str] = None):
    """Create common pipeline components (STT, LLM, TTS, RAG, etc.)."""
    global rag_service, settings

    # STT service (Custom für faster-whisper-server mit expliziter Sprache)
    stt_language = settings.stt_language.lower()
    logger.info(f"STT configured: language={stt_language}, model={settings.stt_model}, url={settings.stt_base_url}")

    stt_service = FasterWhisperSTTService(
        base_url=settings.stt_base_url,
        model=settings.stt_model,
        language=stt_language,
    )

    # LLM service (OpenAI-compatible via vLLM)
    llm_service = OpenAILLMService(
        base_url=f"{settings.llm_base_url}/v1",
        api_key="not-needed",
        model=settings.llm_model,
    )

    # TTS service
    tts_service = HTTPTTSService(
        base_url=settings.tts_base_url,
        sample_rate=24000
    )

    # Context for conversation
    context = LLMContext(
        messages=[
            {"role": "system", "content": settings.system_prompt}
        ]
    )
    context_aggregator = LLMContextAggregatorPair(context)

    # Custom processors
    session_metrics = SessionMetrics()
    rag_processor = RAGProcessor(rag_service, settings, event_manager, session_id)
    rag_processor.set_session_metrics(session_metrics)

    interruption_handler = InterruptionHandler()
    interruption_handler.set_session_metrics(session_metrics)

    response_logger = ResponseLogger(event_manager, session_id)
    response_logger.set_session_metrics(session_metrics)

    sentence_aggregator = SentenceAggregator(tts_service)

    return {
        "stt_service": stt_service,
        "llm_service": llm_service,
        "tts_service": tts_service,
        "context_aggregator": context_aggregator,
        "rag_processor": rag_processor,
        "interruption_handler": interruption_handler,
        "response_logger": response_logger,
        "sentence_aggregator": sentence_aggregator,
        "session_metrics": session_metrics,
    }


async def run_local_bot(webrtc_connection: SmallWebRTCConnection) -> None:
    """Run the voice bot pipeline for a local WebRTC connection."""
    global transport_factory

    components = create_pipeline_components()
    session_metrics = components["session_metrics"]
    logger.info(f"Starting local session {session_metrics.session_id}")

    try:
        # Create transport using factory
        transport = transport_factory.create_local_transport(webrtc_connection)

        # Build pipeline: Audio -> STT -> RAG -> LLM -> TTS -> Audio
        pipeline = Pipeline([
            transport.input(),
            components["stt_service"],           # Transcribe audio to text
            components["rag_processor"],         # Augment with RAG context
            components["context_aggregator"].user(),
            components["llm_service"],
            components["response_logger"],       # Log responses BEFORE TTS conversion
            components["sentence_aggregator"],   # Convert text to TTS audio
            transport.output(),
            components["context_aggregator"].assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)

    except Exception as e:
        logger.error(f"Session {session_metrics.session_id} error: {e}", exc_info=True)
    finally:
        # Cleanup resources
        await components["sentence_aggregator"].cleanup()
        session_metrics.log_summary()
        logger.info(f"Session {session_metrics.session_id} ended")


async def run_daily_bot(room_info: DailyRoomInfo, session_id: str) -> None:
    """Run the voice bot pipeline for a Daily.co room."""
    global transport_factory

    components = create_pipeline_components(session_id=session_id)
    session_metrics = components["session_metrics"]
    logger.info(f"Starting Daily session {session_id} in room {room_info.room_name}")

    # Emit connection event
    await event_manager.emit_connection(session_id, "starting", {"room": room_info.room_name})

    try:
        # Create transport using factory
        transport = transport_factory.create_daily_transport(room_info)

        # Build pipeline: Audio -> STT -> RAG -> LLM -> TTS -> Audio
        pipeline = Pipeline([
            transport.input(),
            components["stt_service"],           # Transcribe audio to text
            components["rag_processor"],         # Augment with RAG context
            components["context_aggregator"].user(),
            components["llm_service"],
            components["response_logger"],       # Log responses BEFORE TTS conversion
            components["sentence_aggregator"],   # Convert text to TTS audio
            transport.output(),
            components["context_aggregator"].assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            )
        )

        await event_manager.emit_connection(session_id, "connected")

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)

    except Exception as e:
        logger.error(f"Session {session_id} error: {e}", exc_info=True)
        await event_manager.emit_error(session_id, str(e), "pipeline")
    finally:
        # Cleanup resources
        await components["sentence_aggregator"].cleanup()
        session_metrics.log_summary()
        await event_manager.emit_connection(session_id, "disconnected")
        logger.info(f"Session {session_id} ended")


async def run_twilio_bot(
    websocket: WebSocket,
    call_info: TwilioCallInfo,
    session_id: str,
) -> None:
    """Run the voice bot pipeline for a Twilio telephone call.

    Uses FastAPIWebsocketTransport with TwilioFrameSerializer.
    Twilio sends mu-law 8kHz audio; the serializer handles conversion.
    """
    global transport_factory

    components = create_pipeline_components(session_id=session_id)
    session_metrics = components["session_metrics"]
    logger.info(
        f"Starting Twilio session {session_id}: "
        f"stream_sid={call_info.stream_sid}, call_sid={call_info.call_sid}"
    )

    await event_manager.emit_connection(session_id, "starting", {
        "stream_sid": call_info.stream_sid,
        "call_sid": call_info.call_sid,
    })

    try:
        transport = transport_factory.create_twilio_transport(
            websocket=websocket,
            call_info=call_info,
        )

        # Build pipeline: Audio -> STT -> RAG -> LLM -> TTS -> Audio
        pipeline = Pipeline([
            transport.input(),
            components["stt_service"],           # Transcribe audio to text
            components["rag_processor"],         # Augment with RAG context
            components["context_aggregator"].user(),
            components["llm_service"],
            components["response_logger"],       # Log responses BEFORE TTS conversion
            components["sentence_aggregator"],   # Convert text to TTS audio
            transport.output(),
            components["context_aggregator"].assistant(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
            )
        )

        await event_manager.emit_connection(session_id, "connected")

        runner = PipelineRunner()
        await runner.run(task)

    except Exception as e:
        logger.error(f"Twilio session {session_id} error: {e}", exc_info=True)
        await event_manager.emit_error(session_id, str(e), "pipeline")
    finally:
        await components["sentence_aggregator"].cleanup()
        session_metrics.log_summary()
        await event_manager.emit_connection(session_id, "disconnected")
        logger.info(f"Twilio session {session_id} ended")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global rag_service, settings, transport_factory, webrtc_handler

    # Startup
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting Orchestrator service...")

    # Initialize transport factory
    transport_config = TransportConfig(
        mode=settings.transport_mode,
        vad_stop_secs=settings.vad_silence_ms / 1000.0,
        daily_api_key=settings.daily_api_key,
        daily_bot_name=settings.daily_bot_name,
        daily_room_expiry=settings.daily_room_expiry_time,
        twilio_account_sid=settings.twilio_account_sid,
        twilio_auth_token=settings.twilio_auth_token,
        twilio_phone_number=settings.twilio_phone_number,
    )
    transport_factory = TransportFactory(transport_config)

    # Initialize WebRTC handler only for local mode
    if transport_factory.is_local:
        ice_servers = transport_factory.get_ice_servers()
        webrtc_handler = SmallWebRTCRequestHandler(ice_servers=ice_servers)
        logger.info(f"WebRTC handler initialized with {len(ice_servers)} ICE servers (STUN only)")

    # Initialize RAG service
    rag_service = RAGService(settings)
    try:
        await rag_service.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        # Continue without RAG - it's optional for POC

    logger.info(f"Orchestrator ready on http://{settings.host}:{settings.port}")
    logger.info(f"Transport mode: {settings.transport_mode.value}")

    yield

    # Shutdown
    logger.info("Shutting down Orchestrator...")
    if rag_service:
        await rag_service.close()
    if transport_factory:
        await transport_factory.close()


# Create FastAPI app
app = FastAPI(
    title="Proof of Concept Voice AI",
    description="Self-hosted voice assistant with STT, LLM, RAG, and TTS",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "orchestrator",
        "transport_mode": settings.transport_mode.value if settings else "unknown",
        "rag_initialized": rag_service._initialized if rag_service else False
    }


# =============================================================================
# System Info Endpoints
# =============================================================================

@app.get("/api/system/info")
async def get_system_info():
    """Return system configuration and model information."""
    import aiohttp

    # Fetch TTS info from service
    tts_info = {"name": "unknown", "sample_rate": 24000}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{settings.tts_base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    tts_info = {
                        "name": data.get("provider", "unknown"),
                        "sample_rate": data.get("sample_rate", 24000),
                        "language": data.get("language", "unknown"),
                    }
    except Exception as e:
        logger.warning(f"Failed to fetch TTS info: {e}")

    return {
        "application": {
            "name": "Proof of Concept Voice AI",
            "version": "0.1.0",
            "transport_mode": settings.transport_mode.value
        },
        "models": {
            "stt": {"name": settings.stt_model, "language": settings.stt_language},
            "llm": {"name": settings.llm_model},
            "tts": tts_info,
            "embedding": {"name": settings.embedding_model}
        },
        "rag": {
            "enabled": rag_service._initialized if rag_service else False,
            "top_k": settings.rag_top_k
        }
    }


@app.get("/api/system/gpu")
async def get_gpu_status():
    """Get GPU status using pynvml directly.

    This approach gives accurate GPU info for all GPUs in a multi-GPU setup,
    since the orchestrator has access to all GPUs for monitoring purposes.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # GPU service assignments from config
        tts_gpu = int(os.environ.get("TTS_GPU_ID", "0"))
        llm_gpu = int(os.environ.get("LLM_GPU_ID", "1"))
        stt_gpu = int(os.environ.get("STT_GPU_ID", "0"))

        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Determine services on this GPU
            services = []
            if i == tts_gpu:
                services.append("tts")
            if i == llm_gpu:
                services.append("llm")
            if i == stt_gpu:
                services.append("stt")

            gpus.append({
                "index": i,
                "name": name if isinstance(name, str) else name.decode(),
                "memory_total_mb": memory.total // (1024 * 1024),
                "memory_used_mb": memory.used // (1024 * 1024),
                "memory_percent": round(memory.used / memory.total * 100, 1),
                "services": services
            })

        pynvml.nvmlShutdown()
        return {
            "status": "available",
            "gpu_count": device_count,
            "gpus": gpus
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU info via pynvml: {e}")
        return {"status": "unavailable", "gpu_count": 0, "gpus": []}


@app.get("/api/system/gpu-config")
async def get_gpu_config():
    """Return GPU configuration for each service."""
    return {
        "stt": {
            "gpu_id": os.environ.get("STT_GPU_ID", "0"),
            "device": os.environ.get("STT_DEVICE", "cuda")
        },
        "llm": {
            "gpu_id": os.environ.get("LLM_GPU_ID", "1"),
        },
        "tts": {
            "gpu_id": os.environ.get("TTS_GPU_ID", "0"),
            "device": os.environ.get("TTS_DEVICE", "cuda")
        }
    }


# =============================================================================
# Daily Mode Endpoints
# =============================================================================

@app.post("/api/session")
async def create_session():
    """Create a new Daily.co session (Daily mode only).

    Returns room URL and token for the client to join.
    """
    if not transport_factory or not transport_factory.is_daily:
        raise HTTPException(
            status_code=400,
            detail="Endpoint only available in Daily mode. Set TRANSPORT_MODE=daily"
        )

    try:
        # Create Daily room
        room_info = await transport_factory.create_daily_room()

        # Use room_name as session_id for event streaming
        session_id = room_info.room_name

        # Start bot in background
        asyncio.create_task(run_daily_bot(room_info, session_id))

        logger.info(f"Created session: {session_id}")

        return {
            "room_url": room_info.room_url,
            "token": room_info.token,
            "room_name": room_info.room_name,
            "session_id": session_id,
        }

    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Twilio Mode Endpoints
# =============================================================================

@app.post("/api/twilio/incoming")
async def twilio_incoming_call(request: Request):
    """Handle incoming Twilio call with TwiML response.

    Twilio calls this webhook when a call comes in.
    Returns TwiML XML that tells Twilio to open a Media Stream WebSocket.
    """
    if not transport_factory or not transport_factory.is_twilio:
        raise HTTPException(
            status_code=400,
            detail="Endpoint only available in Twilio mode. Set TRANSPORT_MODE=twilio"
        )

    # Determine WebSocket URL dynamically (ngrok-compatible)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "localhost:7860")
    # Use wss:// for forwarded (ngrok) connections, ws:// for direct
    scheme = "wss" if request.headers.get("x-forwarded-proto") == "https" else "ws"
    ws_url = f"{scheme}://{host}/ws/twilio"

    logger.info(f"Twilio incoming call -> streaming to {ws_url}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws/twilio")
async def twilio_websocket(websocket: WebSocket):
    """Handle Twilio Media Stream WebSocket connection.

    Twilio connects here after receiving the TwiML <Stream> directive.
    Parses initial messages to extract stream_sid and call_sid,
    then starts the voice bot pipeline.
    """
    if not transport_factory or not transport_factory.is_twilio:
        await websocket.close(code=1008, reason="Twilio mode not enabled")
        return

    await websocket.accept()
    logger.info("Twilio WebSocket accepted")

    stream_sid = None
    call_sid = None

    try:
        # Parse initial Twilio handshake messages (connected + start events)
        for _ in range(2):
            data = await websocket.receive_text()
            msg = json.loads(data)
            event = msg.get("event")

            if event == "connected":
                logger.info("Twilio WebSocket: connected event received")
            elif event == "start":
                start_data = msg.get("start", {})
                stream_sid = msg.get("streamSid", "")
                call_sid = start_data.get("callSid", "")
                logger.info(f"Twilio WebSocket: start event - stream_sid={stream_sid}, call_sid={call_sid}")

        if not stream_sid:
            logger.error("Twilio WebSocket: no streamSid received in handshake")
            await websocket.close(code=1008, reason="Missing streamSid")
            return

        call_info = TwilioCallInfo(stream_sid=stream_sid, call_sid=call_sid or "")
        session_id = f"twilio-{call_sid or stream_sid}"

        # Run the bot pipeline (blocks until call ends)
        await run_twilio_bot(websocket, call_info, session_id)

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Twilio WebSocket error: {e}", exc_info=True)


# =============================================================================
# Local Mode Endpoints
# =============================================================================

@app.get("/api/ice-servers")
async def get_ice_servers_endpoint():
    """Return ICE servers configuration for WebRTC client (local mode only)."""
    if not transport_factory or not transport_factory.is_local:
        raise HTTPException(
            status_code=400,
            detail="Endpoint only available in local mode. Set TRANSPORT_MODE=local"
        )

    return {
        "iceServers": get_local_ice_servers_for_client(),
    }


@app.post("/api/offer")
async def webrtc_offer(request: Request):
    """Handle WebRTC offer from client (local mode only)."""
    if not transport_factory or not transport_factory.is_local:
        raise HTTPException(
            status_code=400,
            detail="Endpoint only available in local mode. Set TRANSPORT_MODE=local"
        )

    try:
        data = await request.json()
        sdp = data.get("sdp")
        sdp_type = data.get("type", "offer")

        if not sdp:
            raise HTTPException(status_code=400, detail="Missing SDP")

        async def on_connection(connection: SmallWebRTCConnection):
            """Callback when WebRTC connection is established."""
            await run_local_bot(connection)

        request_obj = SmallWebRTCRequest(sdp=sdp, type=sdp_type)
        answer = await webrtc_handler.handle_web_request(request_obj, on_connection)

        return JSONResponse(answer)

    except Exception as e:
        logger.error(f"WebRTC offer error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Event Streaming (SSE)
# =============================================================================

@app.get("/api/events/{session_id}")
async def stream_events(session_id: str):
    """Stream real-time events for a session via Server-Sent Events.

    Provides live updates on:
    - Connection status
    - STT transcriptions
    - LLM responses
    - TTS status
    - VAD (voice activity detection)
    - Errors
    """
    async def event_generator():
        queue = await event_manager.subscribe(session_id)
        try:
            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

            while True:
                try:
                    # Wait for events with timeout for keepalive
                    event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield event_data
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await event_manager.unsubscribe(session_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# =============================================================================
# Client Serving
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_client():
    """Serve the appropriate client based on transport mode."""
    if transport_factory and transport_factory.is_daily:
        client_file = CLIENTS_DIR / "daily_client.html"
    elif transport_factory and transport_factory.is_twilio:
        client_file = CLIENTS_DIR / "twilio_status.html"
    else:
        client_file = CLIENTS_DIR / "local_client.html"

    if not client_file.exists():
        raise HTTPException(status_code=500, detail=f"Client file not found: {client_file}")

    return client_file.read_text(encoding="utf-8")


@app.get("/client", response_class=HTMLResponse)
async def get_client_alt():
    """Alternative path for client."""
    return await get_client()


def main():
    """Entry point for the orchestrator."""
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
