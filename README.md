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
| coturn | 3478 (host) | Self-hosted TURN server for WebRTC |
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

# WebRTC / TURN Server
# Self-hosted coturn is included - no external credentials needed
# Default credentials: turnuser / turnpassword (see docker-compose.yml)
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

See **[VAST_AI_DEPLOYMENT.md](VAST_AI_DEPLOYMENT.md)** for detailed instructions.

Quick start:
1. Select a GPU instance (RTX 4090 / 24GB+ VRAM)
2. Clone repo: `git clone <repo-url> && cd PocRealtimeVoiceAI`
3. Configure: `cp .env.example .env`
4. Expose ports in vast.ai console (Direct Port Mappings):
   - `7860/http` (Web UI)
   - `3478/tcp` and `3478/udp` (TURN server)
5. Start: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`
6. Access via the vast.ai assigned URL

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

### Coturn TURN Server Test

Test TURN server connectivity (requires coturn to be running):

```bash
docker compose --profile test-coturn run --rm test-coturn
```

Expected output: Connection statistics showing successful relay through the TURN server.

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - Quick start and overview |
| [VAST_AI_DEPLOYMENT.md](VAST_AI_DEPLOYMENT.md) | Detailed vast.ai deployment guide |
| [AGENTS.md](AGENTS.md) | Comprehensive troubleshooting for AI agents |
| [.env.example](.env.example) | All configuration options |

## License

MIT License - See LICENSE file for details.
