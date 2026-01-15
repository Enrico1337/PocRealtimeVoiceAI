# AGENTS.md - AI Agent Troubleshooting Guide

This document provides comprehensive instructions for AI agents to diagnose and fix issues with the POC Realtime Voice AI system.

## Project Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Browser (WebRTC) ◄──────────────────────────────────────┐     │
│        │                                                   │     │
│        │ Port 7860                          ┌─────────┐   │     │
│        │                                    │ COTURN  │───┘     │
│        ▼                                    │  :3478  │         │
│   ┌─────────────────────────────────────────┴─────────┴─────┐   │
│   │              ORCHESTRATOR (Python/Pipecat)               │   │
│   │  - FastAPI server on :7860                               │   │
│   │  - Serves WebRTC client HTML                             │   │
│   │  - Manages voice pipeline                                │   │
│   │  - RAG retrieval (Qdrant + BGE-M3)                       │   │
│   └─────┬───────────┬───────────┬───────────┬───────────────┘   │
│         │           │           │           │                    │
│         ▼           ▼           ▼           ▼                    │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
│   │   STT   │ │   LLM   │ │   TTS   │ │ Qdrant  │               │
│   │  :8001  │ │  :8002  │ │  :8003  │ │  :6333  │               │
│   │ faster- │ │  vLLM   │ │Chatter- │ │ Vector  │               │
│   │ whisper │ │  Qwen3  │ │  box    │ │   DB    │               │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
/root/PocRealtimeVoiceAI/          # or wherever cloned
├── docker-compose.yml              # Main compose file
├── docker-compose.gpu.yml          # GPU overlay (NVIDIA)
├── .env                            # Runtime config (copy from .env.example)
├── .env.example                    # Template
├── Makefile                        # Convenience commands
├── README.md                       # User documentation
├── AGENTS.md                       # THIS FILE
├── kb/                             # Knowledge base documents
│   └── *.md                        # Markdown files for RAG
├── orchestrator/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # FastAPI + WebRTC entry point
│       ├── pipeline.py             # Pipecat voice pipeline
│       ├── rag.py                  # Qdrant RAG implementation
│       ├── settings.py             # Pydantic settings
│       └── telemetry.py            # Metrics/logging
└── tts/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        ├── __init__.py
        ├── server.py               # FastAPI TTS server
        └── tts_engine.py           # Chatterbox wrapper
```

---

## Diagnostic Commands

### 1. Check Service Status

```bash
# View all container states
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps

# Expected output: All services "healthy" or "running"
# NAME             STATUS          PORTS
# poc-orchestrator healthy         0.0.0.0:7860->7860/tcp
# poc-stt          healthy
# poc-llm          healthy
# poc-tts          healthy
# poc-qdrant       healthy
```

### 2. Check Logs

```bash
# All services
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs --tail=100

# Specific service
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs orchestrator --tail=100
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs llm --tail=100
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs stt --tail=100
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs tts --tail=100
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs qdrant --tail=100

# Follow logs in real-time
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
```

### 3. Check Health Endpoints

```bash
# Orchestrator
curl -s http://localhost:7860/health | jq .

# STT (from inside network)
docker compose exec orchestrator curl -s http://stt:8000/health

# LLM (from inside network)
docker compose exec orchestrator curl -s http://llm:8000/health

# TTS (from inside network)
docker compose exec orchestrator curl -s http://tts:8000/health

# Qdrant (from inside network)
docker compose exec orchestrator curl -s http://qdrant:6333/health
```

### 4. Check GPU Status

```bash
# Host GPU status
nvidia-smi

# GPU inside containers
docker compose exec llm nvidia-smi
docker compose exec stt nvidia-smi
docker compose exec tts nvidia-smi
```

### 5. Check Resource Usage

```bash
# Container resource usage
docker stats --no-stream

# Disk space
df -h

# Memory
free -h
```

### 6. Quick Import Test

```bash
# Validate all pipecat imports without full startup
# Useful after dependency changes or debugging import errors
docker-compose run --rm test

# Expected output:
# Testing pipecat imports...
#   OK: pipecat.audio.vad.silero
#   OK: pipecat.services.tts_service
#   ...
# All imports OK!
```

---

## Common Issues and Solutions

### ISSUE 1: Container won't start - "no such file or directory"

**Symptoms:**
```
Error response from daemon: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: exec: "python": executable file not found in $PATH
```

**Diagnosis:**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs <service> 2>&1 | head -20
```

**Solutions:**

A) Dockerfile CMD issue:
```bash
# Check Dockerfile CMD
cat orchestrator/Dockerfile | grep -A2 "^CMD"
cat tts/Dockerfile | grep -A2 "^CMD"

# Should be:
# CMD ["python", "-m", "app.main"]  (orchestrator)
# CMD ["python", "-m", "app.server"] (tts)
```

B) Rebuild image:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache <service>
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d <service>
```

---

### ISSUE 2: GPU not detected / CUDA errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.is_available() returns False
CUDA error: no kernel image is available for execution on the device
```

**Diagnosis:**
```bash
# Check NVIDIA driver on host
nvidia-smi

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Check GPU in compose
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec llm nvidia-smi
```

**Solutions:**

A) NVIDIA Container Toolkit not installed:
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

B) GPU memory exhausted - reduce LLM memory:
```bash
# Edit .env
LLM_GPU_MEMORY=0.7  # Reduce from 0.85

# Restart
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

C) Wrong CUDA version - check compatibility:
```bash
# Check host CUDA version
nvidia-smi | grep "CUDA Version"

# Container expects CUDA 12.x
# If host has older CUDA, update driver or use CPU mode
```

---

### ISSUE 3: LLM service fails to start / OOM

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
Killed (signal 9)
vLLM worker died
```

**Diagnosis:**
```bash
# Check available GPU memory
nvidia-smi --query-gpu=memory.free --format=csv

# Check LLM logs
docker compose logs llm --tail=50
```

**Solutions:**

A) Reduce model memory usage:
```bash
# Edit .env
LLM_MAX_CONTEXT=16384    # Reduce from 32768
LLM_GPU_MEMORY=0.70      # Reduce from 0.85

# Restart LLM
docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart llm
```

B) Use smaller model:
```bash
# Edit .env - use 4-bit quantized smaller model
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
```

C) Enable CPU offloading (slower but works):
```bash
# In docker-compose.yml, add to llm command:
command: >
  --model ${LLM_MODEL}
  --max-model-len ${LLM_MAX_CONTEXT:-16384}
  --gpu-memory-utilization 0.5
  --cpu-offload-gb 10
  --dtype auto
  --quantization awq
  --trust-remote-code
  --port 8000
```

---

### ISSUE 4: STT not transcribing / wrong language

**Symptoms:**
```
Empty transcription
Wrong language detected
Transcription very slow
```

**Diagnosis:**
```bash
# Check STT logs
docker compose logs stt --tail=50

# Test STT directly
docker compose exec orchestrator curl -X POST http://stt:8000/v1/audio/transcriptions \
  -F "file=@/dev/null" \
  -F "model=Systran/faster-distil-whisper-large-v3" \
  -F "language=de"
```

**Solutions:**

A) Force German language:
```bash
# Verify in .env
STT_LANGUAGE=de

# In orchestrator/app/settings.py, confirm:
stt_language: str = Field(default="de", description="Default language for STT")
```

B) STT model not loaded:
```bash
# Check if model is downloaded
docker compose exec stt ls -la /root/.cache/huggingface/

# Force re-download
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down stt
docker volume rm pocrealtimevoiceai_hf-cache  # Careful: removes all cached models
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d stt
```

C) Audio format issue - check orchestrator sends correct format:
```bash
# In orchestrator/app/main.py, verify audio is 16kHz mono PCM
```

---

### ISSUE 5: TTS not producing audio / Chatterbox errors

**Symptoms:**
```
TTS engine not initialized
Empty audio response
Chatterbox model loading failed
```

**Diagnosis:**
```bash
# Check TTS logs
docker compose logs tts --tail=50

# Test TTS endpoint
docker compose exec orchestrator curl -X POST http://tts:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox","input":"Hallo Welt","voice":"default"}' \
  -o /tmp/test.wav

# Check if file has content
docker compose exec orchestrator ls -la /tmp/test.wav
```

**Solutions:**

A) Chatterbox not installed correctly:
```bash
# Rebuild TTS container
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache tts
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d tts
```

B) GPU memory issue - run TTS on CPU:
```bash
# Edit .env
TTS_DEVICE=cpu

# Restart
docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart tts
```

C) Model download failed:
```bash
# Check HuggingFace cache
docker compose exec tts ls -la /root/.cache/huggingface/hub/

# Clear and retry
docker compose down tts
docker compose up -d tts
# Wait for model download in logs
docker compose logs -f tts
```

---

### ISSUE 6: RAG not working / No context in responses

**Symptoms:**
```
RAG service not initialized
No documents retrieved
Qdrant connection failed
```

**Diagnosis:**
```bash
# Check Qdrant status
docker compose logs qdrant --tail=20

# Check orchestrator RAG initialization
docker compose logs orchestrator | grep -i rag

# Test Qdrant directly
docker compose exec orchestrator curl -s http://qdrant:6333/collections | jq .
```

**Solutions:**

A) Qdrant not ready:
```bash
# Check Qdrant health
docker compose exec orchestrator curl -s http://qdrant:6333/health

# Restart Qdrant
docker compose restart qdrant

# Wait for healthy status
docker compose ps qdrant
```

B) Knowledge base not ingested:
```bash
# Check if kb/ directory has files
ls -la kb/

# Check Qdrant collection
docker compose exec orchestrator curl -s http://qdrant:6333/collections/knowledge_base | jq .

# Force re-ingestion by clearing collection
docker compose exec orchestrator curl -X DELETE http://qdrant:6333/collections/knowledge_base
docker compose restart orchestrator
```

C) Embedding model failed to load:
```bash
# Check orchestrator logs for embedding errors
docker compose logs orchestrator | grep -i embed

# The embedding model runs on CPU, check memory
docker stats poc-orchestrator --no-stream
```

---

### ISSUE 7: WebRTC connection fails

**Symptoms:**
```
Browser shows "Connecting..." indefinitely
ICE connection failed
No audio in/out
```

**Diagnosis:**
```bash
# Check orchestrator logs for WebRTC errors
docker compose logs orchestrator | grep -i webrtc
docker compose logs orchestrator | grep -i ice

# Check if port 7860 is accessible
curl -s http://localhost:7860/health
```

**Solutions:**

A) Port not exposed correctly (vast.ai):
```bash
# Verify port mapping
docker compose ps orchestrator

# On vast.ai: Ensure port 7860 is in "Direct Port Mappings"
# Format: 7860/http
```

B) STUN server not reachable:
```bash
# Test STUN connectivity from container
docker compose exec orchestrator python -c "
import asyncio
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer

async def test():
    config = RTCConfiguration(iceServers=[RTCIceServer(urls='stun:stun.l.google.com:19302')])
    pc = RTCPeerConnection(config)
    offer = await pc.createOffer()
    print('STUN test passed')
    await pc.close()

asyncio.run(test())
"
```

C) Browser microphone not allowed:
```
- Check browser console for permission errors
- Ensure HTTPS or localhost (WebRTC requires secure context)
- On vast.ai: Use the HTTPS URL provided
```

---

### ISSUE 8: Barge-in not working

**Symptoms:**
```
Bot keeps talking when user speaks
No interruption detected
```

**Diagnosis:**
```bash
# Check VAD logs
docker compose logs orchestrator | grep -i vad
docker compose logs orchestrator | grep -i interrupt
docker compose logs orchestrator | grep -i barge
```

**Solutions:**

A) VAD sensitivity too low:
```bash
# Edit .env - reduce silence threshold
VAD_SILENCE_MS=500  # Reduce from 800

# Restart orchestrator
docker compose restart orchestrator
```

B) Pipeline not handling interruptions:
```python
# In orchestrator/app/pipeline.py, verify InterruptionHandler is in pipeline
# Check that StartInterruptionFrame is being sent
```

---

### ISSUE 9: High latency / Slow responses

**Symptoms:**
```
Long delay before bot responds
TTS audio stutters
First word takes >3 seconds
```

**Diagnosis:**
```bash
# Check latency metrics in logs
docker compose logs orchestrator | grep -i latency
docker compose logs orchestrator | grep -i "duration="

# Check GPU utilization
nvidia-smi -l 1
```

**Solutions:**

A) LLM too slow - enable speculative decoding or use faster model:
```bash
# Use faster model variant
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
```

B) TTS chunking not working:
```python
# In orchestrator/app/main.py, verify SentenceAggregator
# is chunking text and sending audio incrementally
```

C) Network latency between containers:
```bash
# Test internal network latency
docker compose exec orchestrator ping -c 5 llm
docker compose exec orchestrator ping -c 5 stt
docker compose exec orchestrator ping -c 5 tts
```

---

### ISSUE 10: Container keeps restarting

**Symptoms:**
```
Container in restart loop
Exit code 137 (OOM killed)
Exit code 1 (application error)
```

**Diagnosis:**
```bash
# Check exit code
docker compose ps -a

# Check logs before crash
docker compose logs <service> --tail=100

# Check system resources
dmesg | tail -50  # Look for OOM killer messages
```

**Solutions:**

A) Exit code 137 (OOM):
```bash
# Increase memory limit or reduce model size
# See ISSUE 3 solutions
```

B) Exit code 1 (Python error):
```bash
# Check logs for Python traceback
docker compose logs <service> 2>&1 | grep -A 20 "Traceback"

# Common fixes:
# - Missing dependencies: rebuild container
# - Config error: check .env values
# - Import error: check requirements.txt
```

---

### ISSUE 11: WebRTC Connection Issues (TURN/ICE)

**Symptoms:**
```
ICE connection state is checking, connection is connecting
Timeout establishing the connection to the remote peer
WebRTC stuck at "Connecting"
```

**Diagnosis:**
```bash
# Check coturn is running and healthy
docker compose ps coturn

# Check coturn logs
docker compose logs coturn

# Check orchestrator TURN configuration
docker compose logs orchestrator | grep -i "coturn\|TURN"

# Test TURN server connectivity
docker compose --profile test-coturn run --rm test-coturn
```

**Cause:**
Self-hosted coturn TURN server not running or not reachable. This happens when:
- Coturn container not started or unhealthy
- Port 3478 not accessible (firewall/vast.ai port mapping)
- Network mode issues

**Solutions:**

A) Verify coturn is running:
```bash
# Start coturn if not running
docker compose up -d coturn

# Check health status
docker compose ps coturn
# Should show "healthy"

# Check logs for errors
docker compose logs coturn --tail=50
```

B) Test TURN server:
```bash
# Run the built-in TURN test
docker compose --profile test-coturn run --rm test-coturn

# Expected output: Connection statistics showing successful relay
```

C) For remote access (vast.ai/cloud):
```bash
# Ensure ports are exposed in cloud console:
# - 7860/http (Web UI)
# - 3478/tcp (TURN TCP)
# - 3478/udp (TURN UDP)

# On vast.ai: Add to "Direct Port Mappings"
```

D) Manual TURN test:
```bash
# Use the WebRTC samples tester:
# https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/
# Add: turn:<your-server-ip>:3478
# Username: turnuser
# Credential: turnpassword
# If "relay" candidates appear, TURN is working
```

---

## Vast.ai Specific Issues

### ISSUE V1: Port not accessible

```bash
# On vast.ai, ports must be explicitly exposed
# In the instance configuration, add:
# Direct Port Mappings: 7860/http

# Or use the auto-assigned port from vast.ai dashboard
# Access via: https://<instance-id>.vast.ai:<mapped-port>/
```

### ISSUE V2: Disk space full

```bash
# Check disk usage
df -h

# Clean Docker cache
docker system prune -a --volumes

# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/hub/*
```

### ISSUE V3: GPU already in use

```bash
# Check what's using GPU
nvidia-smi

# Kill other processes
sudo fuser -v /dev/nvidia*
# Then restart containers
```

### ISSUE V4: Instance terminated unexpectedly

```bash
# Vast.ai may terminate if:
# - Bid price too low (spot instances)
# - Disk quota exceeded
# - Out of credits

# Solution: Use on-demand instance or increase bid
```

---

## Recovery Procedures

### Full System Restart

```bash
cd /path/to/PocRealtimeVoiceAI

# Stop everything
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down

# Clear any stuck resources
docker system prune -f

# Start fresh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Monitor startup
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
```

### Rebuild Single Service

```bash
# Rebuild and restart one service
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache <service>
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d <service>
```

### Factory Reset (Nuclear Option)

```bash
# WARNING: This deletes all data including Qdrant vectors

cd /path/to/PocRealtimeVoiceAI

# Stop and remove everything
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down -v --rmi all

# Remove any leftover volumes
docker volume prune -f

# Rebuild from scratch
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

---

## Log Patterns to Watch For

### Healthy Startup Sequence

```
poc-qdrant       | Qdrant is ready
poc-stt          | Model loaded: Systran/faster-distil-whisper-large-v3
poc-llm          | INFO: Started server process
poc-llm          | INFO: Serving on http://0.0.0.0:8000
poc-tts          | Chatterbox model loaded successfully
poc-orchestrator | RAG service initialized successfully
poc-orchestrator | Orchestrator ready on http://0.0.0.0:7860
```

### Error Patterns

```
# GPU memory error
torch.cuda.OutOfMemoryError: CUDA out of memory

# Model loading failure
OSError: <model> does not appear to have a file named config.json

# Network error
ConnectionRefusedError: [Errno 111] Connection refused

# Timeout
httpx.ReadTimeout: timed out
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL` | `Systran/faster-distil-whisper-large-v3` | Whisper model |
| `STT_LANGUAGE` | `de` | Default STT language |
| `STT_DEVICE` | `cuda` | STT device (cuda/cpu) |
| `LLM_MODEL` | `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ` | LLM model |
| `LLM_MAX_CONTEXT` | `32768` | Max context tokens |
| `LLM_GPU_MEMORY` | `0.85` | GPU memory fraction |
| `TTS_DEVICE` | `cuda` | TTS device (cuda/cpu) |
| `TTS_SAMPLE_RATE` | `24000` | Audio sample rate |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model for RAG |
| `RAG_TOP_K` | `4` | Number of RAG results |
| `VAD_SILENCE_MS` | `800` | Silence before end of utterance |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Quick Fixes Summary

| Problem | Quick Fix |
|---------|-----------|
| OOM | `LLM_GPU_MEMORY=0.6` in .env |
| Slow LLM | Use smaller model or reduce context |
| No GPU | Install nvidia-container-toolkit |
| STT wrong language | Set `STT_LANGUAGE=de` |
| TTS fails | Set `TTS_DEVICE=cpu` |
| RAG empty | Check kb/ has .md files, restart orchestrator |
| Port blocked | Expose 7860 in vast.ai dashboard |
| WebRTC fails | Use HTTPS URL on vast.ai |

---

## Contact / Escalation

If AI agent cannot resolve the issue:
1. Collect all logs: `docker compose logs > debug.log 2>&1`
2. Capture system state: `nvidia-smi > gpu.log && docker ps -a > containers.log`
3. Document steps taken and error messages
4. Escalate to human operator with collected data
