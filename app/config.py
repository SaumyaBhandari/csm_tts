"""
CSM TTS Configuration

Centralized configuration management using environment variables.
"""
import os
from typing import Optional, List
from functools import lru_cache


class Settings:
    """Application settings loaded from environment variables"""
    
    def __init__(self):
        # Server
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        
        # Model
        self.model_device: str = os.getenv("MODEL_DEVICE", "auto")  # auto, cuda, cpu
        
        # Watermark
        self.watermark_enabled: bool = os.getenv("CSM_WATERMARK_ENABLED", "true").lower() == "true"
        self.watermark_key: Optional[List[int]] = self._parse_watermark_key()
        
        # Generation defaults
        self.default_speaker_id: int = int(os.getenv("DEFAULT_SPEAKER_ID", "0"))
        self.default_max_audio_ms: int = int(os.getenv("DEFAULT_MAX_AUDIO_MS", "30000"))
        self.default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.9"))
        self.default_topk: int = int(os.getenv("DEFAULT_TOPK", "50"))
    
    def _parse_watermark_key(self) -> Optional[List[int]]:
        key_str = os.getenv("CSM_WATERMARK_KEY", "")
        if not key_str:
            return None
        try:
            return [int(x.strip()) for x in key_str.split(",")]
        except ValueError:
            return None
    
    def get_device(self) -> str:
        """Determine the device to use for model inference"""
        if self.model_device != "auto":
            return self.model_device
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
