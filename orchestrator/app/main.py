"""
Orchestrator Main: FastAPI server with Pipecat WebRTC pipeline.

Supports two transport modes:
- Daily Mode (default): Uses Daily.co for hosted WebRTC (works with vast.ai)
- Local Mode: Uses SmallWebRTC for direct connections (for LAN/localhost)
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler,
    SmallWebRTCRequest,
)
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TextFrame, Frame, EndFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language

from .pipeline import RAGProcessor, InterruptionHandler, ResponseLogger
from .rag import RAGService
from .settings import Settings, get_settings, TransportMode
from .telemetry import SessionMetrics, setup_logging
from .transport import TransportFactory, TransportConfig, DailyRoomInfo
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
    """

    def __init__(self, tts_service: HTTPTTSService):
        super().__init__()
        self.tts_service = tts_service
        self._buffer = ""
        self._sentence_endings = ".!?"

    async def _generate_tts_audio(self, text: str, direction: FrameDirection):
        """Generate TTS audio for text and push frames.

        Args:
            text: Text to synthesize
            direction: Frame direction for pushing frames
        """
        try:
            async for audio_chunk in self.tts_service.run_tts(text):
                audio_frame = TTSAudioRawFrame(
                    audio=audio_chunk,
                    sample_rate=self.tts_service.sample_rate,
                    num_channels=1
                )
                await self.push_frame(audio_frame, direction)
        except Exception as e:
            logger.error(f"SentenceAggregator TTS error for '{text[:50]}...': {e}")
            # Don't re-raise - allow pipeline to continue without audio

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and frame.text:
            self._buffer += frame.text

            # Check for sentence boundaries
            while self._buffer:
                # Find the first sentence ending
                end_pos = -1
                for char in self._sentence_endings:
                    pos = self._buffer.find(char)
                    if pos != -1 and (end_pos == -1 or pos < end_pos):
                        end_pos = pos

                if end_pos != -1:
                    # Extract sentence
                    sentence = self._buffer[:end_pos + 1].strip()
                    self._buffer = self._buffer[end_pos + 1:].lstrip()

                    if sentence:
                        # Generate TTS for this sentence
                        await self._generate_tts_audio(sentence, direction)
                else:
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


def _get_stt_language():
    """Map language string to Pipecat Language enum."""
    language_map = {
        "de": Language.DE,
        "en": Language.EN,
        "es": Language.ES,
        "fr": Language.FR,
        "it": Language.IT,
        "pt": Language.PT,
        "nl": Language.NL,
        "pl": Language.PL,
        "ru": Language.RU,
        "zh": Language.ZH,
        "ja": Language.JA,
        "ko": Language.KO,
    }
    lang_code = settings.stt_language.lower()
    if lang_code in language_map:
        return language_map[lang_code]
    logger.warning(f"Unknown language '{lang_code}', defaulting to German")
    return Language.DE


def create_pipeline_components(session_id: Optional[str] = None):
    """Create common pipeline components (STT, LLM, TTS, RAG, etc.)."""
    global rag_service, settings

    # STT service (OpenAI-compatible via faster-whisper-server)
    stt_language = _get_stt_language()
    logger.info(f"STT configured: language={stt_language}, model={settings.stt_model}, url={settings.stt_base_url}")

    stt_service = OpenAISTTService(
        api_key="not-needed",
        base_url=f"{settings.stt_base_url}/v1",
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
    return {
        "application": {
            "name": "Proof of Concept Voice AI",
            "version": "0.1.0",
            "transport_mode": settings.transport_mode.value
        },
        "models": {
            "stt": {"name": settings.stt_model, "language": settings.stt_language},
            "llm": {"name": settings.llm_model},
            "tts": {"name": "chatterbox", "sample_rate": 24000},
            "embedding": {"name": settings.embedding_model}
        },
        "rag": {
            "enabled": rag_service._initialized if rag_service else False,
            "top_k": settings.rag_top_k
        }
    }


@app.get("/api/system/gpu")
async def get_gpu_status():
    """Aggregate GPU status from TTS service."""
    import aiohttp
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(f"{settings.tts_base_url}/gpu") as response:
                if response.status == 200:
                    return await response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch GPU info: {e}")
    return {"status": "unavailable", "gpu_count": 0, "gpus": []}


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
    """Serve the appropriate WebRTC client based on transport mode."""
    if transport_factory and transport_factory.is_daily:
        client_file = CLIENTS_DIR / "daily_client.html"
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
