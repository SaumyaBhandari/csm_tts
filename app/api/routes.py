"""
TTS API Routes

REST API endpoints for text-to-speech generation.
"""
import base64
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.config import Settings, get_settings
from app.core import get_tts_service, TTSService
from app.models import TTSRequest, TTSResponse, HealthResponse

router = APIRouter()


# ============================================================================
# Dependencies
# ============================================================================

def get_service() -> TTSService:
    """Dependency to get TTS service"""
    return get_tts_service()


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    service: Annotated[TTSService, Depends(get_service)],
    settings: Annotated[Settings, Depends(get_settings)]
):
    """
    Health check endpoint.
    
    Returns server status and model state.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=service.is_loaded,
        device=service.device,
        version="1.0.0"
    )


@router.post("/generate", response_model=TTSResponse, tags=["TTS"])
async def generate_speech(
    request: TTSRequest,
    service: Annotated[TTSService, Depends(get_service)],
    settings: Annotated[Settings, Depends(get_settings)]
):
    """
    Generate speech from text.
    
    Returns base64-encoded WAV audio.
    
    - **text**: Text to convert to speech (required)
    - **speaker_id**: Speaker voice ID (0 or 1)
    - **max_audio_length_ms**: Maximum audio length in ms
    - **temperature**: Sampling temperature (0.1-2.0)
    - **topk**: Top-k sampling parameter
    - **watermark**: Enable/disable watermark
    - **watermark_key**: Custom watermark key (5 integers)
    """
    start_time = time.time()
    
    # Resolve watermark settings
    enable_watermark = request.watermark if request.watermark is not None else settings.watermark_enabled
    watermark_key = request.watermark_key or settings.watermark_key
    
    try:
        wav_bytes, sample_rate, duration_ms = service.generate_wav_bytes(
            text=request.text,
            speaker_id=request.speaker_id,
            max_audio_length_ms=request.max_audio_length_ms,
            temperature=request.temperature,
            topk=request.topk,
            enable_watermark=enable_watermark,
            watermark_key=watermark_key
        )
        
        return TTSResponse(
            audio_base64=base64.b64encode(wav_bytes).decode(),
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            processing_time_ms=(time.time() - start_time) * 1000,
            watermarked=enable_watermark
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/wav", tags=["TTS"])
async def generate_speech_wav(
    request: TTSRequest,
    service: Annotated[TTSService, Depends(get_service)],
    settings: Annotated[Settings, Depends(get_settings)]
):
    """
    Generate speech from text.
    
    Returns raw WAV audio file.
    """
    # Resolve watermark settings
    enable_watermark = request.watermark if request.watermark is not None else settings.watermark_enabled
    watermark_key = request.watermark_key or settings.watermark_key
    
    try:
        wav_bytes, sample_rate, _ = service.generate_wav_bytes(
            text=request.text,
            speaker_id=request.speaker_id,
            max_audio_length_ms=request.max_audio_length_ms,
            temperature=request.temperature,
            topk=request.topk,
            enable_watermark=enable_watermark,
            watermark_key=watermark_key
        )
        
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
