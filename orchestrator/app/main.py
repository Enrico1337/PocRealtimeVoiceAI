"""
Orchestrator Main: FastAPI server with Pipecat WebRTC pipeline.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler,
    SmallWebRTCRequest,
)
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from aiortc import RTCIceServer
from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TextFrame, Frame, EndFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .pipeline import RAGProcessor, InterruptionHandler, ResponseLogger
from .rag import RAGService
from .settings import Settings, get_settings
from .telemetry import SessionMetrics, setup_logging

logger = logging.getLogger(__name__)

# Global instances
rag_service: Optional[RAGService] = None
settings: Optional[Settings] = None
webrtc_handler: Optional[SmallWebRTCRequestHandler] = None


def get_ice_servers() -> list:
    """Build ICE server list from settings."""
    servers = [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
    ]
    if settings and settings.turn_username and settings.turn_credential:
        # Metered.ca TURN servers - exact URLs from dashboard
        servers.extend([
            RTCIceServer(urls="stun:stun.relay.metered.ca:80"),
            RTCIceServer(
                urls="turn:global.relay.metered.ca:80",
                username=settings.turn_username,
                credential=settings.turn_credential,
            ),
            RTCIceServer(
                urls="turn:global.relay.metered.ca:80?transport=tcp",
                username=settings.turn_username,
                credential=settings.turn_credential,
            ),
            RTCIceServer(
                urls="turn:global.relay.metered.ca:443",
                username=settings.turn_username,
                credential=settings.turn_credential,
            ),
            RTCIceServer(
                urls="turns:global.relay.metered.ca:443?transport=tcp",
                username=settings.turn_username,
                credential=settings.turn_credential,
            ),
        ])
    return servers


def get_ice_servers_for_client() -> list:
    """Get ICE servers in format suitable for JavaScript client."""
    servers = [
        {"urls": "stun:stun.l.google.com:19302"},
    ]
    if settings and settings.turn_username and settings.turn_credential:
        # Metered.ca TURN servers - exact URLs from dashboard
        servers.extend([
            {"urls": "stun:stun.relay.metered.ca:80"},
            {
                "urls": "turn:global.relay.metered.ca:80",
                "username": settings.turn_username,
                "credential": settings.turn_credential,
            },
            {
                "urls": "turn:global.relay.metered.ca:80?transport=tcp",
                "username": settings.turn_username,
                "credential": settings.turn_credential,
            },
            {
                "urls": "turn:global.relay.metered.ca:443",
                "username": settings.turn_username,
                "credential": settings.turn_credential,
            },
            {
                "urls": "turns:global.relay.metered.ca:443?transport=tcp",
                "username": settings.turn_username,
                "credential": settings.turn_credential,
            },
        ])
    return servers


class HTTPTTSService(TTSService):
    """TTS Service that calls our custom HTTP TTS endpoint."""

    def __init__(self, base_url: str, sample_rate: int = 24000):
        super().__init__(sample_rate=sample_rate)
        self.base_url = base_url.rstrip("/")
        self._client = None

    async def start(self, frame: Frame):
        import httpx
        self._client = httpx.AsyncClient(timeout=60.0)
        await super().start(frame)

    async def stop(self, frame: Frame):
        if self._client:
            await self._client.aclose()
        await super().stop(frame)

    async def run_tts(self, text: str):
        """Generate speech from text and yield audio frames."""
        if not text.strip():
            return

        try:
            response = await self._client.post(
                f"{self.base_url}/v1/audio/speech",
                json={
                    "model": "chatterbox",
                    "input": text,
                    "voice": "default",
                    "response_format": "pcm",
                    "speed": 1.0
                }
            )
            response.raise_for_status()

            # Yield audio in chunks
            audio_data = response.content
            chunk_size = 4800  # 100ms at 24kHz, 16-bit
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk

        except Exception as e:
            logger.error(f"TTS error: {e}")


class SentenceAggregator(FrameProcessor):
    """Aggregates text into sentences for TTS."""

    def __init__(self, tts_service: HTTPTTSService):
        super().__init__()
        self.tts_service = tts_service
        self._buffer = ""
        self._sentence_endings = ".!?"

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
                        async for audio_chunk in self.tts_service.run_tts(sentence):
                            from pipecat.frames.frames import AudioRawFrame
                            audio_frame = AudioRawFrame(
                                audio=audio_chunk,
                                sample_rate=self.tts_service.sample_rate,
                                num_channels=1
                            )
                            await self.push_frame(audio_frame, direction)
                else:
                    break

        elif isinstance(frame, EndFrame):
            # Flush remaining buffer
            if self._buffer.strip():
                async for audio_chunk in self.tts_service.run_tts(self._buffer.strip()):
                    from pipecat.frames.frames import AudioRawFrame
                    audio_frame = AudioRawFrame(
                        audio=audio_chunk,
                        sample_rate=self.tts_service.sample_rate,
                        num_channels=1
                    )
                    await self.push_frame(audio_frame, direction)
                self._buffer = ""
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


async def run_bot(webrtc_connection: SmallWebRTCConnection) -> None:
    """Run the voice bot pipeline for a WebRTC connection."""
    global rag_service, settings

    session_metrics = SessionMetrics()
    logger.info(f"Starting session {session_metrics.session_id}")

    try:
        # Create transport
        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        stop_secs=settings.vad_silence_ms / 1000.0,
                    )
                ),
                audio_in_passthrough=True,
            )
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
        rag_processor = RAGProcessor(rag_service, settings)
        rag_processor.set_session_metrics(session_metrics)

        interruption_handler = InterruptionHandler()
        interruption_handler.set_session_metrics(session_metrics)

        response_logger = ResponseLogger()
        response_logger.set_session_metrics(session_metrics)

        sentence_aggregator = SentenceAggregator(tts_service)

        # Build pipeline
        pipeline = Pipeline([
            transport.input(),
            rag_processor,
            context_aggregator.user(),
            llm_service,
            sentence_aggregator,
            response_logger,
            transport.output(),
            context_aggregator.assistant(),
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
        session_metrics.log_summary()
        logger.info(f"Session {session_metrics.session_id} ended")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global rag_service, settings, webrtc_handler

    # Startup
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting Orchestrator service...")

    # Initialize WebRTC handler with ICE servers from settings
    webrtc_handler = SmallWebRTCRequestHandler(ice_servers=get_ice_servers())
    logger.info(f"WebRTC handler initialized with TURN server: {settings.turn_server_url}")

    # Initialize RAG service
    rag_service = RAGService(settings)
    try:
        await rag_service.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        # Continue without RAG - it's optional for POC

    logger.info(f"Orchestrator ready on http://{settings.host}:{settings.port}")

    yield

    # Shutdown
    logger.info("Shutting down Orchestrator...")
    if rag_service:
        await rag_service.close()


# Create FastAPI app
app = FastAPI(
    title="POC Realtime Voice AI",
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
        "rag_initialized": rag_service._initialized if rag_service else False
    }


@app.get("/api/ice-servers")
async def get_ice_servers_endpoint():
    """Return ICE servers configuration for WebRTC client."""
    return {"iceServers": get_ice_servers_for_client()}


@app.post("/api/offer")
async def webrtc_offer(request: Request):
    """Handle WebRTC offer from client."""
    try:
        data = await request.json()
        sdp = data.get("sdp")
        sdp_type = data.get("type", "offer")

        if not sdp:
            raise HTTPException(status_code=400, detail="Missing SDP")

        async def on_connection(connection: SmallWebRTCConnection):
            """Callback when WebRTC connection is established."""
            await run_bot(connection)

        request_obj = SmallWebRTCRequest(sdp=sdp, type=sdp_type)
        answer = await webrtc_handler.handle_web_request(request_obj, on_connection)

        return JSONResponse(answer)

    except Exception as e:
        logger.error(f"WebRTC offer error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Serve simple WebRTC client
CLIENT_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>POC Voice AI</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #fff;
        }
        .container {
            text-align: center;
            padding: 2rem;
            max-width: 500px;
        }
        h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: #e94560;
        }
        .subtitle {
            color: #8892b0;
            margin-bottom: 2rem;
        }
        .status {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        .status-indicator.disconnected { background: #ff6b6b; }
        .status-indicator.connecting { background: #ffd93d; }
        .status-indicator.connected { background: #6bcb77; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        button {
            background: #e94560;
            color: white;
            border: none;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 0.5rem;
        }
        button:hover:not(:disabled) {
            background: #ff6b6b;
            transform: translateY(-2px);
        }
        button:disabled {
            background: #4a4a6a;
            cursor: not-allowed;
        }
        .transcript {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 2rem;
            max-height: 300px;
            overflow-y: auto;
            text-align: left;
        }
        .transcript-entry {
            padding: 0.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .transcript-entry.user { color: #6bcb77; }
        .transcript-entry.bot { color: #64b5f6; }
        .mic-active {
            animation: mic-pulse 1s infinite;
        }
        @keyframes mic-pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(233, 69, 96, 0.4); }
            50% { box-shadow: 0 0 0 20px rgba(233, 69, 96, 0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>POC Voice AI</h1>
        <p class="subtitle">Self-hosted Realtime Voice Assistant</p>

        <div class="status">
            <span class="status-indicator disconnected" id="statusIndicator"></span>
            <span id="statusText">Disconnected</span>
        </div>

        <button id="connectBtn" onclick="connect()">Connect</button>
        <button id="disconnectBtn" onclick="disconnect()" disabled>Disconnect</button>

        <div class="transcript" id="transcript">
            <div class="transcript-entry" style="color: #8892b0;">
                Click "Connect" and start speaking...
            </div>
        </div>
    </div>

    <script>
        let pc = null;
        let localStream = null;
        let audioContext = null;

        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const connectBtn = document.getElementById('connectBtn');
        const disconnectBtn = document.getElementById('disconnectBtn');
        const transcript = document.getElementById('transcript');

        function setStatus(status, text) {
            statusIndicator.className = 'status-indicator ' + status;
            statusText.textContent = text;
        }

        function addTranscript(text, type) {
            const entry = document.createElement('div');
            entry.className = 'transcript-entry ' + type;
            entry.textContent = (type === 'user' ? 'You: ' : 'Bot: ') + text;
            transcript.appendChild(entry);
            transcript.scrollTop = transcript.scrollHeight;
        }

        async function connect() {
            try {
                setStatus('connecting', 'Connecting...');
                connectBtn.disabled = true;

                // Fetch ICE servers from server
                const iceResponse = await fetch('/api/ice-servers');
                if (!iceResponse.ok) {
                    throw new Error('Failed to fetch ICE servers');
                }
                const iceConfig = await iceResponse.json();
                console.log('ICE servers:', iceConfig.iceServers);

                // Get microphone access
                localStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    }
                });

                // Create peer connection with ICE servers from server
                pc = new RTCPeerConnection({
                    iceServers: iceConfig.iceServers,
                    iceTransportPolicy: 'all'
                });

                // Add local audio track
                localStream.getTracks().forEach(track => {
                    pc.addTrack(track, localStream);
                });

                // Handle remote audio
                pc.ontrack = (event) => {
                    console.log('Received remote track');
                    const audio = new Audio();
                    audio.srcObject = event.streams[0];
                    audio.play().catch(e => console.log('Audio play error:', e));
                };

                // Create and send offer
                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                // Wait for ICE gathering
                await new Promise((resolve) => {
                    if (pc.iceGatheringState === 'complete') {
                        resolve();
                    } else {
                        pc.onicegatheringstatechange = () => {
                            if (pc.iceGatheringState === 'complete') {
                                resolve();
                            }
                        };
                    }
                });

                // Send offer to server
                const response = await fetch('/api/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });

                if (!response.ok) {
                    throw new Error('Server error: ' + response.status);
                }

                const answer = await response.json();
                await pc.setRemoteDescription(new RTCSessionDescription(answer));

                setStatus('connected', 'Connected - Speak now!');
                connectBtn.disabled = true;
                disconnectBtn.disabled = false;
                connectBtn.classList.add('mic-active');

                transcript.innerHTML = '<div class="transcript-entry" style="color: #6bcb77;">Connected! Start speaking...</div>';

            } catch (error) {
                console.error('Connection error:', error);
                setStatus('disconnected', 'Error: ' + error.message);
                connectBtn.disabled = false;
                disconnect();
            }
        }

        function disconnect() {
            if (pc) {
                pc.close();
                pc = null;
            }
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }

            setStatus('disconnected', 'Disconnected');
            connectBtn.disabled = false;
            disconnectBtn.disabled = true;
            connectBtn.classList.remove('mic-active');
        }

        // Handle page unload
        window.addEventListener('beforeunload', disconnect);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_client():
    """Serve the WebRTC client."""
    return CLIENT_HTML


@app.get("/client", response_class=HTMLResponse)
async def get_client_alt():
    """Alternative path for client."""
    return CLIENT_HTML


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
