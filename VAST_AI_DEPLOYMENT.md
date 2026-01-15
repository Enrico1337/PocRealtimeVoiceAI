# Vast.ai Deployment Guide

Schritt-für-Schritt Anleitung zum Deployment des POC Realtime Voice AI auf vast.ai.

## Voraussetzungen

- vast.ai Account mit Guthaben
- Das Repository auf GitHub (oder als ZIP verfügbar)

---

## Schritt 1: Instance auswählen

### Empfohlene Spezifikationen

| Komponente | Minimum | Empfohlen |
|------------|---------|-----------|
| GPU | RTX 3090 (24GB) | RTX 4090 (24GB) |
| VRAM | 24 GB | 24+ GB |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB | 150 GB |
| CUDA | 12.0+ | 12.1+ |

### Instance-Suche auf vast.ai

1. Gehe zu [vast.ai/console/create](https://vast.ai/console/create)
2. Filter setzen:
   - GPU: `RTX 4090` oder `RTX 3090`
   - VRAM: >= 24 GB
   - Disk Space: >= 100 GB
   - CUDA Version: >= 12.0
3. Sortiere nach: "$/hr" (günstigster zuerst)
4. Wähle eine Instance mit guter "DLP Score" (Reliability)

### Instance Template

Wähle als Template:
```
pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

Oder nutze ein leeres Template und Docker.

---

## Schritt 2: Instance konfigurieren

### Wichtige Einstellungen

**Docker Options:**
```
- Expose Port: 7860/http
- Docker Compose: enabled (falls verfügbar)
```

**Environment Variables** (optional, können auch in .env gesetzt werden):
```bash
HF_TOKEN=<dein-huggingface-token>  # Falls nötig für gated models
```

**On-start Script** (optional):
```bash
#!/bin/bash
cd /root
git clone https://github.com/<your-repo>/PocRealtimeVoiceAI.git
cd PocRealtimeVoiceAI
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## Schritt 3: Mit Instance verbinden

### Option A: Web Terminal

1. Klicke auf "Open" bei deiner Instance
2. Wähle "Terminal" oder "JupyterLab Terminal"

### Option B: SSH

```bash
# SSH-Befehl von vast.ai Dashboard kopieren
ssh -p <port> root@<ip-address> -L 7860:localhost:7860
```

Die `-L 7860:localhost:7860` Option erstellt einen SSH-Tunnel für lokalen Zugriff.

---

## Schritt 4: Repository klonen und starten

```bash
# In das Home-Verzeichnis wechseln
cd /root

# Repository klonen
git clone https://github.com/<your-repo>/PocRealtimeVoiceAI.git
cd PocRealtimeVoiceAI

# Konfiguration erstellen
cp .env.example .env

# Optional: .env anpassen
nano .env

# Services starten
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Logs verfolgen (Strg+C zum Beenden)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
```

### Startup-Dauer

Die erste Ausführung dauert länger wegen Model-Downloads:
- Qdrant: ~1 Minute
- STT (Whisper): ~5-10 Minuten (Model ~3GB)
- LLM (Qwen3-30B): ~10-20 Minuten (Model ~15GB)
- TTS (Chatterbox): ~5-10 Minuten (Model ~2GB)
- Orchestrator + RAG Embeddings: ~5-10 Minuten

**Gesamtdauer erster Start: 20-45 Minuten**

Nachfolgende Starts sind deutlich schneller (~2-5 Minuten), da Models gecached sind.

---

## Schritt 5: Port-Zugriff einrichten

### Option A: Direct Port Mapping (empfohlen)

1. Im vast.ai Dashboard: Klicke auf deine Instance
2. Gehe zu "Edit" oder "Configure"
3. Unter "Direct Port Mappings" hinzufügen:
   ```
   7860/http
   ```
4. Speichern und warten bis Instance neu konfiguriert

5. Zugriff über:
   ```
   https://<instance-id>-7860.proxy.vast.ai/
   ```
   oder die angezeigte URL im Dashboard

### Option B: SSH Tunnel (für Debugging)

```bash
# Von deinem lokalen Rechner:
ssh -p <vast-ssh-port> root@<vast-ip> -L 7860:localhost:7860

# Dann im Browser:
http://localhost:7860
```

### Option C: Cloudflare Tunnel (permanent)

```bash
# Auf der vast.ai Instance:
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared.deb

# Quick Tunnel erstellen (temporär)
cloudflared tunnel --url http://localhost:7860

# Gibt eine URL wie: https://random-name.trycloudflare.com
```

---

## Schritt 6: Testen

### Health Check

```bash
# Auf der Instance
curl http://localhost:7860/health

# Erwartete Antwort:
# {"status":"healthy","service":"orchestrator","rag_initialized":true}
```

### Browser Test

1. Öffne die URL (je nach Port-Methode):
   - `https://<instance-id>-7860.proxy.vast.ai/`
   - oder `http://localhost:7860` (mit SSH Tunnel)

2. Klicke "Connect"

3. Erlaube Mikrofon-Zugriff

4. Sprich: "Wie beantrage ich einen Personalausweis?"

5. Die KI sollte mit Sprache antworten

---

## Troubleshooting auf vast.ai

### Problem: Services starten nicht

```bash
# Logs prüfen
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs

# GPU verfügbar?
nvidia-smi

# Docker läuft?
docker ps
```

### Problem: Out of Memory

```bash
# GPU Memory prüfen
nvidia-smi

# .env anpassen
nano .env
# Ändern:
LLM_GPU_MEMORY=0.7
LLM_MAX_CONTEXT=16384

# Neu starten
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Problem: Port nicht erreichbar

```bash
# Port prüfen
netstat -tlnp | grep 7860

# Firewall?
ufw status

# Container läuft?
docker compose ps
```

### Problem: Langsame Downloads

```bash
# HuggingFace Mirror setzen
export HF_ENDPOINT=https://hf-mirror.com

# In .env hinzufügen für Container:
# HF_ENDPOINT=https://hf-mirror.com
```

### Problem: Instance wird beendet (Spot)

- Spot-Instances können jederzeit beendet werden
- Lösung: On-Demand Instance verwenden (teurer aber stabil)
- Oder: Höheres Bid setzen für Spot

---

## Kosten-Optimierung

### GPU Auswahl

| GPU | VRAM | ca. Preis/Stunde | Empfehlung |
|-----|------|------------------|------------|
| RTX 3090 | 24GB | $0.20-0.40 | Budget |
| RTX 4090 | 24GB | $0.40-0.80 | Standard |
| A100 40GB | 40GB | $1.00-2.00 | Premium |

### Spot vs On-Demand

- **Spot** (Interruptible): 50-70% günstiger, kann jederzeit beendet werden
- **On-Demand**: Garantierte Verfügbarkeit, teurer

### Disk Space

- Minimiere auf 100GB wenn möglich
- Models werden in Docker Volumes gecached
- Mehr Disk = höhere Kosten

---

## Automatisches Startup-Script

Erstelle `/root/start-voice-ai.sh`:

```bash
#!/bin/bash
set -e

cd /root/PocRealtimeVoiceAI

# Git pull für Updates
git pull origin main 2>/dev/null || true

# Sicherstellen dass .env existiert
if [ ! -f .env ]; then
    cp .env.example .env
fi

# Container starten
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Auf Healthy warten
echo "Waiting for services to be healthy..."
sleep 30

# Status prüfen
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps

echo ""
echo "========================================"
echo "Voice AI is starting up!"
echo "This may take 10-20 minutes on first run."
echo ""
echo "Monitor logs with:"
echo "docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f"
echo ""
echo "Access at: http://localhost:7860"
echo "========================================"
```

```bash
chmod +x /root/start-voice-ai.sh
```

---

## Persistente Daten

### Was wird gecached

- HuggingFace Models: `/root/.cache/huggingface/` (in Docker Volume `hf-cache`)
- Qdrant Vektoren: Docker Volume `qdrant-data`

### Backup vor Instance-Stop

```bash
# Optional: Models sichern (falls Instance gelöscht wird)
docker run --rm -v pocrealtimevoiceai_hf-cache:/data -v /root/backup:/backup alpine tar cvf /backup/hf-cache.tar /data
```

---

## Quick Reference

### Starten
```bash
cd /root/PocRealtimeVoiceAI
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Stoppen
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

### Logs
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f
```

### Status
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml ps
nvidia-smi
```

### Neustart einzelner Service
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml restart orchestrator
```

### Komplett neu bauen
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```
