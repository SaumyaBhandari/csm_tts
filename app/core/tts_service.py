"""
TTS Service

Core text-to-speech generation logic using CSM model.
"""
import io
import time
from typing import Optional, List, Tuple

import torch
import torchaudio

from app.config import get_settings


class TTSService:
    """
    Text-to-Speech service wrapping the CSM model.
    
    Implements singleton pattern - model loaded once on first use.
    """
    
    _instance: Optional["TTSService"] = None
    _generator = None
    _load_time: Optional[float] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def is_loaded(self) -> bool:
        return self._generator is not None
    
    @property
    def device(self) -> str:
        if self._generator:
            return str(self._generator.device)
        return "not loaded"
    
    @property
    def sample_rate(self) -> int:
        if self._generator:
            return self._generator.sample_rate
        return 24000  # Default CSM sample rate
    
    def load_model(self) -> None:
        """Load the CSM model into memory"""
        if self._generator is not None:
            return
        
        import os
        os.environ["NO_TORCH_COMPILE"] = "1"
        
        print("Loading CSM-1B model...")
        start = time.time()
        
        # Import generator from project root
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from generator import load_csm_1b
        
        settings = get_settings()
        device = settings.get_device()
        
        self._generator = load_csm_1b(device)
        self._load_time = time.time() - start
        
        print(f"Model loaded in {self._load_time:.1f}s on {device}")
    
    def generate(
        self,
        text: str,
        speaker_id: int = 0,
        max_audio_length_ms: int = 30000,
        temperature: float = 0.9,
        topk: int = 50,
        enable_watermark: bool = True,
        watermark_key: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            speaker_id: Speaker voice ID (0 or 1)
            max_audio_length_ms: Maximum audio length
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            enable_watermark: Whether to apply watermark
            watermark_key: Custom watermark key
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        self.load_model()
        
        audio = self._generator.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            enable_watermark=enable_watermark,
            watermark_key=watermark_key,
        )
        
        return audio, self._generator.sample_rate
    
    def generate_wav_bytes(
        self,
        text: str,
        speaker_id: int = 0,
        max_audio_length_ms: int = 30000,
        temperature: float = 0.9,
        topk: int = 50,
        enable_watermark: bool = True,
        watermark_key: Optional[List[int]] = None
    ) -> Tuple[bytes, int, float]:
        """
        Generate speech and return as WAV bytes.
        
        Returns:
            Tuple of (wav_bytes, sample_rate, duration_ms)
        """
        audio, sample_rate = self.generate(
            text=text,
            speaker_id=speaker_id,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
            enable_watermark=enable_watermark,
            watermark_key=watermark_key
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), sample_rate, format="wav")
        buffer.seek(0)
        
        duration_ms = (len(audio) / sample_rate) * 1000
        
        return buffer.read(), sample_rate, duration_ms


# Singleton instance
def get_tts_service() -> TTSService:
    """Get the TTS service singleton"""
    return TTSService()
