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
    exaggeration: float = 0.5
    cfg_weight: float = 0.5

    # Coqui-specific
    coqui_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    coqui_speaker_wav: Optional[str] = None
