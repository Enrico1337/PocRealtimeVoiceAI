"""
Pipecat Voice Pipeline: STT -> LLM (+RAG) -> TTS with VAD and Barge-in.
"""

import asyncio
import logging
from typing import Optional

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai import OpenAILLMService

from .rag import RAGService
from .settings import Settings
from .telemetry import SessionMetrics, track_latency

logger = logging.getLogger(__name__)


class RAGProcessor(FrameProcessor):
    """Processor that augments user messages with RAG context."""

    def __init__(self, rag_service: RAGService, settings: Settings):
        super().__init__()
        self.rag_service = rag_service
        self.settings = settings
        self.session_metrics: Optional[SessionMetrics] = None

    def set_session_metrics(self, metrics: SessionMetrics) -> None:
        self.session_metrics = metrics

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            # Retrieve relevant context
            user_text = frame.text
            logger.info(f"[{self.session_metrics.session_id if self.session_metrics else 'unknown'}] "
                       f"User: {user_text}")

            context_text = ""
            if self.rag_service._initialized:
                try:
                    if self.session_metrics:
                        with track_latency("rag", self.session_metrics):
                            docs = await self.rag_service.retrieve(user_text)
                    else:
                        docs = await self.rag_service.retrieve(user_text)

                    context_text = self.rag_service.format_context(docs)
                    if context_text:
                        logger.debug(f"RAG context retrieved from {len(docs)} documents")

                except Exception as e:
                    logger.error(f"RAG retrieval error: {e}")

            # Augment the transcription with context
            if context_text:
                augmented_text = (
                    f"Kontext aus der Wissensbasis:\n{context_text}\n\n"
                    f"Benutzeranfrage: {user_text}"
                )
                frame = TranscriptionFrame(
                    text=augmented_text,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp
                )

        await self.push_frame(frame, direction)


class InterruptionHandler(FrameProcessor):
    """Handler for barge-in / interruption events."""

    def __init__(self):
        super().__init__()
        self.session_metrics: Optional[SessionMetrics] = None
        self._is_speaking = False

    def set_session_metrics(self, metrics: SessionMetrics) -> None:
        self.session_metrics = metrics

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._is_speaking:
                # Barge-in detected
                if self.session_metrics:
                    self.session_metrics.barge_in_count += 1
                logger.info(
                    f"[{self.session_metrics.session_id if self.session_metrics else 'unknown'}] "
                    "Barge-in detected - interrupting bot"
                )
                # Send interruption frame to stop TTS
                await self.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            pass  # VAD handles this

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._is_speaking = False

        elif isinstance(frame, TextFrame):
            self._is_speaking = True

        await self.push_frame(frame, direction)


class ResponseLogger(FrameProcessor):
    """Log bot responses for debugging."""

    def __init__(self):
        super().__init__()
        self.session_metrics: Optional[SessionMetrics] = None
        self._current_response = []

    def set_session_metrics(self, metrics: SessionMetrics) -> None:
        self.session_metrics = metrics

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and frame.text:
            self._current_response.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._current_response:
                full_response = "".join(self._current_response)
                logger.info(
                    f"[{self.session_metrics.session_id if self.session_metrics else 'unknown'}] "
                    f"Bot: {full_response}"
                )
                self._current_response = []

            if self.session_metrics:
                self.session_metrics.turn_count += 1

        await self.push_frame(frame, direction)


class OpenAICompatibleSTTService:
    """STT service using OpenAI-compatible API (faster-whisper-server)."""

    def __init__(self, base_url: str, model: str, language: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language = language

    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files={"file": ("audio.wav", audio_data, "audio/wav")},
                data={
                    "model": self.model,
                    "language": self.language,
                    "response_format": "json"
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("text", "")


class OpenAICompatibleTTSService:
    """TTS service using OpenAI-compatible API."""

    def __init__(self, base_url: str, sample_rate: int = 24000):
        self.base_url = base_url.rstrip("/")
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/v1/audio/speech",
                json={
                    "model": "chatterbox",
                    "input": text,
                    "voice": "default",
                    "response_format": "wav",
                    "speed": 1.0
                }
            )
            response.raise_for_status()
            return response.content


async def create_pipeline(
    transport,
    settings: Settings,
    rag_service: RAGService,
    session_metrics: SessionMetrics
) -> PipelineTask:
    """Create the voice pipeline with all processors."""

    # VAD analyzer for detecting speech
    vad_analyzer = SileroVADAnalyzer(
        params=SileroVADAnalyzer.VADParams(
            min_silence_duration_ms=settings.vad_silence_ms,
        )
    )

    # LLM service (OpenAI-compatible via vLLM)
    llm_service = OpenAILLMService(
        base_url=f"{settings.llm_base_url}/v1",
        api_key="not-needed",  # vLLM doesn't require API key
        model=settings.llm_model,
    )

    # Context for conversation
    context = OpenAILLMContext(
        messages=[
            {"role": "system", "content": settings.system_prompt}
        ]
    )
    context_aggregator = llm_service.create_context_aggregator(context)

    # Custom processors
    rag_processor = RAGProcessor(rag_service, settings)
    rag_processor.set_session_metrics(session_metrics)

    interruption_handler = InterruptionHandler()
    interruption_handler.set_session_metrics(session_metrics)

    response_logger = ResponseLogger()
    response_logger.set_session_metrics(session_metrics)

    # Build pipeline
    pipeline = Pipeline([
        transport.input(),
        vad_analyzer,
        interruption_handler,
        rag_processor,
        context_aggregator.user(),
        llm_service,
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

    return task
