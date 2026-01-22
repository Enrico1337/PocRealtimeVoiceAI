"""
Chatterbox TTS Engine implementation.
"""

import logging

import numpy as np
import torch

from .base import TTSEngine
from .types import TTSConfig, TTSProvider

logger = logging.getLogger(__name__)


class ChatterboxEngine(TTSEngine):
    """TTS engine using Chatterbox model."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self.model = None
        self._is_multilingual = False
        self.exaggeration = config.exaggeration
        self.cfg_weight = config.cfg_weight

        # Check CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

    @property
    def provider(self) -> TTSProvider:
        return TTSProvider.CHATTERBOX

    def initialize(self) -> None:
        """Load the Chatterbox model (Multilingual for non-English)."""
        if self._initialized:
            return

        logger.info(f"Loading Chatterbox model on {self.device} for language: {self.language}")

        try:
            if self.language == "en":
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                self._is_multilingual = False
            else:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                self._is_multilingual = True

            self._initialized = True
            logger.info(f"Chatterbox {'Multilingual ' if self._is_multilingual else ''}model loaded successfully")

        except ImportError as e:
            logger.error(f"Chatterbox import error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
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
        """Generate audio using Chatterbox model."""
        generate_kwargs = {
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
        }
        if self._is_multilingual:
            generate_kwargs["language_id"] = self.language

        wav = self.model.generate(text, **generate_kwargs)

        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        return wav
