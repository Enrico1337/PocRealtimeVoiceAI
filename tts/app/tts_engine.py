"""
DEPRECATED: Legacy TTS Engine wrapper.

This module is deprecated and will be removed in a future version.
Use `from app.engines import get_engine, ChatterboxEngine` instead.
"""

import warnings

# Re-export for backward compatibility
from .engines import ChatterboxEngine, get_engine

warnings.warn(
    "tts_engine module is deprecated. Use 'from app.engines import get_engine' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["ChatterboxEngine", "get_engine"]
