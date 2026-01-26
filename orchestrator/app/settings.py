"""
Pydantic Settings for the Orchestrator service.
All configuration via environment variables.
"""

from enum import Enum
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class TransportMode(str, Enum):
    """Transport mode for WebRTC."""
    DAILY = "daily"
    LOCAL = "local"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # =========================================================================
    # Transport Mode
    # =========================================================================
    transport_mode: TransportMode = Field(
        default=TransportMode.DAILY,
        description="'daily' for Daily.co hosted WebRTC or 'local' for SmallWebRTC"
    )

    # =========================================================================
    # Daily.co Configuration (only used when transport_mode=daily)
    # =========================================================================
    daily_api_key: str = Field(
        default="",
        description="Daily.co API key (from https://dashboard.daily.co/developers)"
    )
    daily_bot_name: str = Field(
        default="Voice Assistant",
        description="Bot display name in Daily room"
    )
    daily_room_expiry_time: int = Field(
        default=3600,
        description="Room expiry time in seconds"
    )

    # =========================================================================
    # Service URLs
    # =========================================================================
    stt_base_url: str = Field(default="http://stt:8000", description="STT service base URL")
    llm_base_url: str = Field(default="http://llm:8000", description="LLM service base URL")
    tts_base_url: str = Field(default="http://tts:8000", description="TTS service base URL")
    qdrant_host: str = Field(default="qdrant", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")

    # =========================================================================
    # Model Configuration
    # =========================================================================
    stt_model: str = Field(
        default="Systran/faster-whisper-large-v3",
        description="Whisper model for STT"
    )
    stt_language: str = Field(default="de", description="Default language for STT")
    llm_model: str = Field(
        default="stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ",
        description="LLM model name"
    )
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Embedding model for RAG"
    )

    # =========================================================================
    # Pipeline Configuration
    # =========================================================================
    vad_silence_ms: int = Field(
        default=800,
        description="Silence duration (ms) to finalize utterance"
    )
    rag_top_k: int = Field(default=4, description="Number of RAG results to retrieve")
    rag_collection_name: str = Field(default="knowledge_base", description="Qdrant collection name")
    kb_path: str = Field(default="/app/kb", description="Knowledge base directory path")
    chunk_size: int = Field(default=512, description="Text chunk size for RAG")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")

    # =========================================================================
    # Server Configuration
    # =========================================================================
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=7860, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # =========================================================================
    # Timeouts (in seconds)
    # =========================================================================
    stt_timeout: float = Field(default=30.0, description="STT request timeout")
    llm_timeout: float = Field(default=60.0, description="LLM request timeout")
    tts_timeout: float = Field(default=30.0, description="TTS request timeout")

    # =========================================================================
    # System Prompt
    # =========================================================================
    system_prompt: str = Field(
        default="""Du bist ein hilfreicher Sprach-Assistent für Behördenanfragen.

Deine Aufgaben:
- Antworte kurz, präzise und freundlich
- Nutze den bereitgestellten Kontext um Fragen zu beantworten
- Wenn du etwas nicht weißt, sage es ehrlich
- Sprich natürlich, als würdest du mit jemandem telefonieren
- Vermeide lange Aufzählungen, fasse zusammen

Formatierung für Sprachausgabe:
- KEINE Markdown-Formatierung (kein **, *, _, #)
- KEINE Klammern, URLs oder Sonderzeichen
- Zahlen unter 13 als Worte (zum Beispiel "drei" statt "3")
- Abkürzungen ausschreiben (zum Beispiel "zum Beispiel" statt "z.B.")
- Jeder Satz endet mit Punkt, Fragezeichen oder Ausrufezeichen

Wichtig: Halte deine Antworten kurz (2-3 Sätze), da sie vorgelesen werden.""",
        description="System prompt for the LLM"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
