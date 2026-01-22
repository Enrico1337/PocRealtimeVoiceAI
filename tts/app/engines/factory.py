"""
TTS Engine Factory.
"""

import logging
import os
from typing import Optional

from .base import TTSEngine
from .types import TTSConfig, TTSProvider

logger = logging.getLogger(__name__)

# Global engine instance
_engine: Optional[TTSEngine] = None


class TTSEngineFactory:
    """Factory for creating TTS engine instances."""

    @staticmethod
    def create(config: TTSConfig) -> TTSEngine:
        """
        Create a TTS engine based on the configuration.

        Args:
            config: TTS configuration

        Returns:
            TTSEngine instance

        Raises:
            ValueError: If provider is not supported
        """
        if config.provider == TTSProvider.CHATTERBOX:
            from .chatterbox_engine import ChatterboxEngine
            return ChatterboxEngine(config)

        elif config.provider == TTSProvider.COQUI:
            from .coqui_engine import CoquiEngine
            return CoquiEngine(config)

        else:
            raise ValueError(f"Unsupported TTS provider: {config.provider}")


def get_config_from_env() -> TTSConfig:
    """Build TTSConfig from environment variables."""
    provider_str = os.environ.get("TTS_PROVIDER", "chatterbox").lower()
    try:
        provider = TTSProvider(provider_str)
    except ValueError:
        logger.warning(f"Unknown TTS_PROVIDER '{provider_str}', defaulting to chatterbox")
        provider = TTSProvider.CHATTERBOX

    return TTSConfig(
        provider=provider,
        device=os.environ.get("TTS_DEVICE", "cuda"),
        sample_rate=int(os.environ.get("TTS_SAMPLE_RATE", "24000")),
        language=os.environ.get("TTS_LANGUAGE", "de"),
        exaggeration=float(os.environ.get("TTS_EXAGGERATION", "0.5")),
        cfg_weight=float(os.environ.get("TTS_CFG_WEIGHT", "0.5")),
        coqui_model=os.environ.get("TTS_COQUI_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"),
        coqui_speaker_wav=os.environ.get("TTS_COQUI_SPEAKER_WAV") or None,
    )


def get_engine() -> TTSEngine:
    """Get or create the global TTS engine from environment configuration."""
    global _engine
    if _engine is None:
        config = get_config_from_env()
        logger.info(f"Creating TTS engine: provider={config.provider.value}")
        _engine = TTSEngineFactory.create(config)
    return _engine


def reset_engine() -> None:
    """Reset the global engine instance (useful for testing)."""
    global _engine
    _engine = None
