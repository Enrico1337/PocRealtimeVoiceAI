"""
TTS Server: OpenAI-compatible API for Chatterbox TTS.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .tts_engine import get_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""
    model: str = Field(default="chatterbox", description="Model to use")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Voice to use")
    response_format: Literal["wav", "pcm", "mp3"] = Field(
        default="wav",
        description="Audio format"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup."""
    logger.info("Starting TTS service...")

    # Initialize engine
    engine = get_engine()
    engine.initialize()

    logger.info(f"TTS service ready (device: {engine.device}, sample_rate: {engine.sample_rate})")

    yield

    logger.info("TTS service shutting down")


app = FastAPI(
    title="Chatterbox TTS Service",
    description="OpenAI-compatible TTS API using Chatterbox",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    engine = get_engine()
    return {
        "status": "healthy" if engine.is_initialized else "initializing",
        "service": "tts",
        "device": engine.device,
        "sample_rate": engine.sample_rate
    }


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    OpenAI-compatible speech synthesis endpoint.

    Returns audio in the requested format.
    """
    engine = get_engine()

    if not engine.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="TTS engine not initialized"
        )

    if not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text is empty"
        )

    start_time = time.perf_counter()

    try:
        if request.response_format == "pcm":
            audio_bytes = engine.synthesize_pcm(request.input)
            media_type = "audio/pcm"
        else:
            # Default to WAV
            audio_bytes = engine.synthesize(request.input)
            media_type = "audio/wav"

        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Synthesized {len(request.input)} chars in {duration_ms:.0f}ms "
            f"({len(audio_bytes)} bytes)"
        )

        return Response(
            content=audio_bytes,
            media_type=media_type,
            headers={
                "X-Synthesis-Time-Ms": str(int(duration_ms))
            }
        )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {str(e)}"
        )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatibility)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "chatterbox",
                "object": "model",
                "created": 1700000000,
                "owned_by": "resemble-ai"
            }
        ]
    }


@app.get("/gpu")
async def get_gpu_info():
    """Return GPU information via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index": i,
                "name": name if isinstance(name, str) else name.decode(),
                "memory_total_mb": memory.total // (1024 * 1024),
                "memory_used_mb": memory.used // (1024 * 1024),
                "memory_percent": round(memory.used / memory.total * 100, 1)
            })
        pynvml.nvmlShutdown()
        return {"status": "ok", "gpu_count": device_count, "gpus": gpus}
    except Exception as e:
        return {"status": "error", "message": str(e), "gpu_count": 0, "gpus": []}


def main():
    """Entry point."""
    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "8000"))

    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
