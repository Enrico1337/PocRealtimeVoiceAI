"""
Abstract base class for TTS engines.
"""

import io
import logging
import wave
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .types import TTSConfig, TTSProvider

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Abstract base class for TTS engines."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.language = config.language
        self.device = config.device
        self._initialized = False

    @property
    @abstractmethod
    def provider(self) -> TTSProvider:
        """Return the provider type."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Load the model and prepare for synthesis."""
        pass

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to WAV audio bytes.

        Args:
            text: Text to synthesize

        Returns:
            WAV audio as bytes
        """
        pass

    @abstractmethod
    def synthesize_pcm(self, text: str) -> bytes:
        """
        Synthesize text to raw PCM audio bytes (16-bit, mono).

        Args:
            text: Text to synthesize

        Returns:
            Raw PCM audio as bytes
        """
        pass

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy array to WAV bytes."""
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

        return buffer.getvalue()

    def _empty_wav(self) -> bytes:
        """Generate empty WAV file."""
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"")

        return buffer.getvalue()

    def _normalize_audio(self, wav: np.ndarray) -> np.ndarray:
        """Normalize audio to int16 format."""
        if wav.ndim > 1:
            wav = wav.squeeze()
        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767).astype(np.int16)
