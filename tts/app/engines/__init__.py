"""
TTS Engines Package.

Provides a factory-based approach for multiple TTS providers.
"""

from .base import TTSEngine
from .chatterbox_engine import ChatterboxEngine
from .coqui_engine import CoquiEngine
from .factory import TTSEngineFactory, get_config_from_env, get_engine, reset_engine
from .types import TTSConfig, TTSProvider

__all__ = [
    "TTSEngine",
    "TTSProvider",
    "TTSConfig",
    "TTSEngineFactory",
    "ChatterboxEngine",
    "CoquiEngine",
    "get_engine",
    "get_config_from_env",
    "reset_engine",
]
