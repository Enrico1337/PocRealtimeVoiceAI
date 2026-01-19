# POC Realtime Voice AI

Self-hosted Proof-of-Concept for a realtime voice assistant with:
- **STT**: faster-whisper (OpenAI-compatible API)
- **LLM**: vLLM with Qwen3-30B-A3B (AWQ quantized)
- **RAG**: Qdrant + BGE-M3 embeddings
- **TTS**: Chatterbox Turbo

```
Browser (WebRTC) → STT → LLM (+RAG) → TTS → Browser
```

## 5-Minute Quickstart

### Prerequisites
- Docker + Docker Compose
- NVIDIA GPU (24GB+ VRAM recommended)
- NVIDIA Container Toolkit installed

### 1. Clone & Configure

```bash
# Clone repository
git clone <repo-url>
cd PocRealtimeVoiceAI

# Copy environment config
cp .env.example .env

# Optional: Edit .env to customize settings
```

### 2. Start Services

```bash
# Build and start with GPU support
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Or use the Makefile:
```bash
make build
make up
```

### 3. Connect

Open your browser: **http://localhost:7860**

Click "Connect" and start speaking!

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser                                 │
│                    (WebRTC Client)                              │
└───────────────────────┬─────────────────────────────────────────┘
                        │ WebRTC
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestrator                                │
│                   (Pipecat Pipeline)                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │   VAD   │ →  │   STT   │ →  │ LLM+RAG │ →  │   TTS   │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
└───────┬─────────────┬─────────────┬─────────────┬───────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Silero  │   │ faster- │   │  vLLM   │   │Chatter- │
   │  VAD    │   │ whisper │   │ +Qdrant │   │  box    │
   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| orchestrator | 7860 (exposed) | Main pipeline + Web UI |
| stt | 8001 (exposed) | faster-whisper-server |
| llm | 8002 (exposed) | vLLM OpenAI server |
| tts | 8003 (exposed) | Chatterbox TTS |
| qdrant | 6333 (exposed) | Vector database |

## Configuration

Key environment variables (see `.env.example`):

```bash
# Models
STT_MODEL=Systran/faster-distil-whisper-large-v3
LLM_MODEL=stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ
EMBEDDING_MODEL=BAAI/bge-m3

# Pipeline
STT_LANGUAGE=de              # Default language for STT
VAD_SILENCE_MS=800          # Silence before finalizing utterance
RAG_TOP_K=4                 # Number of RAG results

# Resources
LLM_MAX_CONTEXT=32768       # Context window
LLM_GPU_MEMORY=0.85         # GPU memory utilization

# Transport Mode
TRANSPORT_MODE=daily        # "daily" (default) or "local"
DAILY_API_KEY=              # Required for cloud deployment (vast.ai)
```

## Knowledge Base

Place your documents in the `./kb` directory:
- Supported formats: `.md`, `.txt`, `.rst`
- Documents are automatically chunked and embedded on startup
- Example files included for a German government services bot

## Features

- **Barge-in**: Interrupt the bot by speaking while it's responding
- **VAD Turn-Taking**: Automatic speech detection with configurable silence threshold
- **Streaming TTS**: Audio starts playing before full response is generated
- **RAG Context**: Relevant knowledge base excerpts are injected into prompts

## Cloud Deployment (vast.ai)

With Daily.co transport, cloud deployment requires only port 7860 (no TURN server needed).

1. Get Daily.co API key from [dashboard.daily.co](https://dashboard.daily.co)
2. Select a GPU instance (RTX 4090 / 24GB+ VRAM)
3. Clone and configure:
   ```bash
   git clone <repo-url> && cd PocRealtimeVoiceAI
   cp .env.example .env
   # Set in .env:
   TRANSPORT_MODE=daily
   DAILY_API_KEY=your-api-key
   ```
4. Expose port `7860/http` in vast.ai console
5. Start: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
6. Access via the vast.ai assigned URL

## vast.ai Multi-GPU Deployment

For instances with multiple GPUs (e.g., 2x RTX 4090), use the Multi-GPU override to distribute services optimally.

### GPU Assignment

| Service | GPU | VRAM Usage |
|---------|-----|------------|
| STT     | 0   | ~3-4 GB    |
| TTS     | 0   | ~2-3 GB    |
| LLM     | 1   | ~20-24 GB  |

### Start with Multi-GPU

```bash
# Configure .env
cp .env.example .env
# Adjust GPU IDs if needed (STT_GPU_ID, TTS_GPU_ID, LLM_GPU_ID)

# Start with multi-GPU override
docker compose -f docker-compose.yml -f docker-compose.vast.yml up -d
```

### Verify GPU Assignment

```bash
# Check GPU allocation per container
docker exec poc-stt nvidia-smi
docker exec poc-tts nvidia-smi
docker exec poc-llm nvidia-smi

# Check GPU config via API
curl http://localhost:7860/api/system/gpu-config
```

## Troubleshooting

For comprehensive troubleshooting, see **[AGENTS.md](AGENTS.md)** - a detailed guide for diagnosing and fixing issues (also useful for AI agents).

### Services won't start
```bash
# Check logs
make logs

# Check specific service
docker compose logs stt
docker compose logs llm
```

### Out of GPU memory
- Reduce `LLM_GPU_MEMORY` in `.env`
- Use a smaller model
- Reduce `LLM_MAX_CONTEXT`

### STT not recognizing speech
- Check microphone permissions in browser
- Verify STT language setting matches your speech
- Check STT container logs

### No audio output
- Ensure browser allows audio autoplay
- Check TTS container logs
- Verify audio output device is working

## Health Checks

```bash
# Check orchestrator health
curl http://localhost:7860/health

# Or use Makefile
make health
```

## DSGVO / Privacy Notes

This POC is designed for self-hosted, EU-based deployment:
- **No audio recording**: Audio is processed in memory only
- **No external APIs**: All processing happens locally
- **Content logging**: Enabled for debugging (transcripts, prompts visible in logs)
- **Telemetry**: Only latency metrics and session IDs (random UUIDs)

For production use, review logging settings and implement appropriate data handling.

## Development

```bash
# Rebuild specific service
make rebuild-orchestrator
make rebuild-tts

# Open shell in container
make shell-orchestrator

# Clean up everything
make clean
```

## Testing

### Quick Import Test

Validate all pipecat imports before deployment (useful after dependency changes):

```bash
docker compose --profile test run --rm test
```

Expected output:
```
Testing pipecat imports...
  OK: pipecat.audio.vad.silero
  OK: pipecat.services.tts_service
  ...
All imports OK!
```

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - Quick start and overview |
| [AGENTS.md](AGENTS.md) | Comprehensive troubleshooting for AI agents |
| [.env.example](.env.example) | All configuration options |

## License

MIT License - See LICENSE file for details.
