"""
Coqui TTS Engine implementation using XTTS-v2.
"""

import logging
from typing import Optional

import numpy as np
import torch

from .base import TTSEngine
from .types import TTSConfig, TTSProvider

logger = logging.getLogger(__name__)


class CoquiEngine(TTSEngine):
    """TTS engine using Coqui XTTS-v2 model."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.model = None
        self.model_name = config.coqui_model
        self.speaker_wav = config.coqui_speaker_wav

        # Check CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

    @property
    def provider(self) -> TTSProvider:
        return TTSProvider.COQUI

    def initialize(self) -> None:
        """Load the Coqui XTTS-v2 model."""
        if self._initialized:
            return

        logger.info(f"Loading Coqui TTS model '{self.model_name}' on {self.device}")

        try:
            from TTS.api import TTS

            self.model = TTS(
                model_name=self.model_name,
                gpu=(self.device == "cuda")
            )

            self._initialized = True
            logger.info(f"Coqui TTS model loaded successfully (language: {self.language})")

        except ImportError as e:
            logger.error(f"Coqui TTS import error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS: {e}")
            raise

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to WAV audio bytes.

        Args:
            text: Text to synthesize

        Returns:
            WAV audio as bytes
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if not text.strip():
            return self._empty_wav()

        try:
            wav = self._generate(text)
            wav_int16 = self._normalize_audio(wav)
            return self._to_wav_bytes(wav_int16)

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self._empty_wav()

    def synthesize_pcm(self, text: str) -> bytes:
        """
        Synthesize text to raw PCM audio bytes (16-bit, mono).

        Args:
            text: Text to synthesize

        Returns:
            Raw PCM audio as bytes
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        if not text.strip():
            return b""

        try:
            wav = self._generate(text)
            wav_int16 = self._normalize_audio(wav)
            return wav_int16.tobytes()

        except Exception as e:
            logger.error(f"PCM synthesis error: {e}")
            return b""

    def _generate(self, text: str) -> np.ndarray:
        """Generate audio using Coqui XTTS-v2 model."""
        tts_kwargs = {
            "text": text,
            "language": self.language,
        }

        # Add speaker_wav for voice cloning if configured
        if self.speaker_wav:
            tts_kwargs["speaker_wav"] = self.speaker_wav

        wav = self.model.tts(**tts_kwargs)

        # Convert list to numpy array if needed
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)

        return wav
