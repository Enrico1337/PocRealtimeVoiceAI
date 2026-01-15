"""
Chatterbox TTS Engine wrapper.
Loads the model once and provides synthesis methods.
"""

import io
import logging
import os
import wave
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ChatterboxEngine:
    """Wrapper for Chatterbox TTS model."""

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 24000
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.sample_rate = sample_rate
        self.model = None
        self._initialized = False

        if self.device == "cpu" and device == "cuda":
            logger.warning("CUDA not available, falling back to CPU")

    def initialize(self) -> None:
        """Load the Chatterbox model."""
        if self._initialized:
            return

        logger.info(f"Loading Chatterbox model on {self.device}...")

        try:
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_pretrained(
                device=self.device
            )
            self._initialized = True
            logger.info("Chatterbox model loaded successfully")

        except ImportError:
            logger.error(
                "Chatterbox not installed. Install with: "
                "pip install chatterbox-tts"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
            raise

    def synthesize(
        self,
        text: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5
    ) -> bytes:
        """
        Synthesize text to WAV audio bytes.

        Args:
            text: Text to synthesize
            exaggeration: Emotion/expression level (0-1)
            cfg_weight: Classifier-free guidance weight

        Returns:
            WAV audio as bytes
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if not text.strip():
            return self._empty_wav()

        try:
            # Generate audio
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )

            # Convert to numpy array
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            # Ensure correct shape
            if wav.ndim > 1:
                wav = wav.squeeze()

            # Normalize to int16
            wav = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav * 32767).astype(np.int16)

            # Create WAV bytes
            return self._to_wav_bytes(wav_int16)

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self._empty_wav()

    def synthesize_pcm(
        self,
        text: str,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5
    ) -> bytes:
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
            wav = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )

            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()

            if wav.ndim > 1:
                wav = wav.squeeze()

            wav = np.clip(wav, -1.0, 1.0)
            wav_int16 = (wav * 32767).astype(np.int16)

            return wav_int16.tobytes()

        except Exception as e:
            logger.error(f"PCM synthesis error: {e}")
            return b""

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

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# Global engine instance
_engine: Optional[ChatterboxEngine] = None


def get_engine() -> ChatterboxEngine:
    """Get or create the global TTS engine."""
    global _engine
    if _engine is None:
        device = os.environ.get("TTS_DEVICE", "cuda")
        sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
        _engine = ChatterboxEngine(device=device, sample_rate=sample_rate)
    return _engine
