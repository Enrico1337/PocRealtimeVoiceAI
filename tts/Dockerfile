# TTS Service Dockerfile
# Multi-provider TTS with CUDA support (Chatterbox + Coqui XTTS-v2)

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Avoid interactive prompts (tzdata etc.)
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

# Install system dependencies
# espeak-ng: Required for Coqui TTS phonemization
# libsndfile1-dev: Required for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    build-essential \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    espeak-ng \
    git \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Build argument for TTS provider selection
ARG TTS_BUILD_PROVIDER=chatterbox

# Install Python dependencies based on selected provider
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    if [ "$TTS_BUILD_PROVIDER" = "coqui" ]; then \
        pip install --no-build-isolation coqui-tts>=0.27.3 && \
        pip install torchcodec; \
    else \
        pip install --no-build-isolation chatterbox-tts>=0.1.6; \
    fi

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Environment defaults
# COQUI_TOS_AGREED=1 accepts the Coqui Public Model License (CPML) for non-commercial use
# This skips the interactive license prompt that would fail in Docker (no stdin)
ENV TTS_PROVIDER=chatterbox \
    TTS_DEVICE=cuda \
    TTS_SAMPLE_RATE=24000 \
    TTS_LANGUAGE=de \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8000 \
    COQUI_TOS_AGREED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "app.server"]
