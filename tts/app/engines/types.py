"""
TTS Engine Types and Configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TTSProvider(str, Enum):
    """Available TTS providers."""
    CHATTERBOX = "chatterbox"
    COQUI = "coqui"


@dataclass
class TTSConfig:
    """Configuration for TTS engines."""
    provider: TTSProvider = TTSProvider.CHATTERBOX
    device: str = "cuda"
    sample_rate: int = 24000
    language: str = "de"

    # Chatterbox-specific
    # Lower values (0.3) improve stability for German speech
    exaggeration: float = 0.3
    cfg_weight: float = 0.3

    # Coqui-specific
    coqui_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    coqui_speaker_wav: Optional[str] = None
    coqui_speaker: str = "Claribel Dervla"  # Default preset speaker
