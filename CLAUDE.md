# CLAUDE.md - AI Agent Guidelines

## Arbeitsweise

- Halte dich strikt an bestehende Projekt-Konventionen (Architektur, Naming, Lint/Formatter, Patterns).
- Schreibe wartbaren, idiomatischen Code: kleine, klare Änderungen > große Refactors.
- Vermeide Overengineering: implementiere nur das, was gefordert ist oder eindeutig nötig.

### APIs / SDKs: keine veralteten Methoden

Wenn du eine API/SDK verwendest oder eine Implementierungs-Variante auswählst:

- Ermittle zuerst die **tatsächlich verwendete Version** im Repo (z.B. `*.csproj`, `Directory.Packages.props`, `package.json`, Lockfiles).
- Prüfe **online** die **offizielle** Doku + Changelogs/Migration Guides für diese Version (oder die aktuelle stabile, falls Version unklar).
- Prüfe auch **online** die **offizielle** Doku + Changelogs/Migration Guides für diese Version wenn du neue API oder SDKs einbindest.
- Nutze **keine** deprecated/obsolete APIs oder alten Beispiele; bevorzuge den **empfohlenen** Weg aus der aktuellen Doku.
- Wenn es mehrere Wege gibt: wähle den empfohlenen und begründe kurz.

### Qualität & Abschluss

- Ergänze/aktualisiere Tests, wenn Verhalten geändert wird.
- Führe Build/Lint/Tests aus (oder liefere exakt die Befehle, die ich ausführen soll).
- Kein "Drive-by Refactor": keine Neben-Verbesserungen ohne explizite Anforderung.
- Markiere Annahmen klar und nenne relevante Doku-Links, die du genutzt hast (kurz).
- **Nach erfolgreicher Implementierung**: Änderungen committen und auf Git pushen mit verständlichem Commit-Kommentar.

---

## Projekt-Kontext

Dieses Projekt ist ein Self-hosted Realtime Voice AI System mit folgenden Komponenten:

| Service | Technologie | Port |
|---------|-------------|------|
| Orchestrator | Python/Pipecat/FastAPI | 7860 |
| STT | faster-whisper-server | 8001 |
| LLM | vLLM (Qwen3) | 8002 |
| TTS | Chatterbox / Coqui XTTS-v2 | 8003 |
| Vector DB | Qdrant | 6333 |

### Wichtige Dateien

- `docker-compose.yml` - Haupt-Compose-Datei
- `docker-compose.vast.yml` - Multi-GPU Override
- `.env.example` - Alle Konfigurationsoptionen
- `orchestrator/app/` - Voice Pipeline Code
- `tts/app/engines/` - TTS Provider (Factory Pattern)

### Code-Konventionen

- Python: Type Hints, Pydantic für Settings/Models
- Docker: Multi-Stage Builds, Health Checks
- Logging: Strukturiert mit Service-Prefix
- Patterns: Factory Pattern für austauschbare Komponenten (TTS, Transport)
