"""
Request/Response Schemas

Pydantic models for API request and response validation.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# Request Schemas
# ============================================================================

class TTSRequest(BaseModel):
    """Request body for TTS generation"""
    
    text: str = Field(
        ...,
        description="Text to convert to speech",
        min_length=1,
        max_length=5000,
        examples=["Hello, world!"]
    )
    
    speaker_id: int = Field(
        default=0,
        ge=0,
        description="Speaker ID (0 or 1 for different voices)"
    )
    
    max_audio_length_ms: int = Field(
        default=30000,
        ge=1000,
        le=90000,
        description="Maximum audio length in milliseconds"
    )
    
    temperature: float = Field(
        default=0.9,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more variation)"
    )
    
    topk: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Top-k sampling parameter"
    )
    
    watermark: Optional[bool] = Field(
        default=None,
        description="Enable/disable watermark. If None, uses server default."
    )
    
    watermark_key: Optional[List[int]] = Field(
        default=None,
        description="Custom watermark key (list of 5 integers)"
    )


# ============================================================================
# Response Schemas
# ============================================================================

class TTSResponse(BaseModel):
    """Response body for TTS generation"""
    
    audio_base64: str = Field(
        description="Base64-encoded WAV audio"
    )
    
    sample_rate: int = Field(
        description="Audio sample rate in Hz"
    )
    
    duration_ms: float = Field(
        description="Audio duration in milliseconds"
    )
    
    processing_time_ms: float = Field(
        description="Server processing time in milliseconds"
    )
    
    watermarked: bool = Field(
        description="Whether audio has watermark applied"
    )


class HealthResponse(BaseModel):
    """Response body for health check"""
    
    status: str = Field(
        description="Server status"
    )
    
    model_loaded: bool = Field(
        description="Whether the model is loaded in memory"
    )
    
    device: str = Field(
        description="Compute device (cuda/cpu)"
    )
    
    version: str = Field(
        description="API version"
    )


class ErrorResponse(BaseModel):
    """Error response body"""
    
    error: str = Field(
        description="Error message"
    )
    
    detail: Optional[str] = Field(
        default=None,
        description="Detailed error information"
    )
