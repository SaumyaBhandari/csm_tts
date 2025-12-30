"""
CSM TTS API Server

FastAPI server that exposes the CSM (Conversational Speech Model) as an HTTP API.
Designed for serverless GPU deployment (auto-scale, warm-start on request).

Environment Variables:
    CSM_WATERMARK_KEY: Private watermark key as comma-separated integers (e.g., "1,2,3,4,5")
    CSM_WATERMARK_ENABLED: Default watermark state ("true" or "false", default: "true")
"""
import os
import io
import base64
import time
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Disable Triton compilation for serverless compatibility
os.environ["NO_TORCH_COMPILE"] = "1"

# Load environment configuration
def get_watermark_key_from_env() -> Optional[List[int]]:
    """Load private watermark key from environment variable."""
    key_str = os.environ.get("CSM_WATERMARK_KEY", "")
    if not key_str:
        return None
    try:
        return [int(x.strip()) for x in key_str.split(",")]
    except ValueError:
        print(f"WARNING: Invalid CSM_WATERMARK_KEY format: {key_str}")
        return None

def get_default_watermark_enabled() -> bool:
    """Get default watermark enabled state from environment."""
    return os.environ.get("CSM_WATERMARK_ENABLED", "true").lower() == "true"

# Environment-based configuration
ENV_WATERMARK_KEY = get_watermark_key_from_env()
ENV_WATERMARK_ENABLED = get_default_watermark_enabled()

# Global generator instance (warm-start after first load)
generator = None
last_request_time = None


class TTSRequest(BaseModel):
    """Request model for TTS generation"""
    text: str
    speaker_id: int = 0
    max_audio_length_ms: int = 30000
    temperature: float = 0.9
    topk: int = 50
    # Watermark options
    watermark: Optional[bool] = Field(
        default=None, 
        description="Enable/disable watermark. If not provided, uses server default (CSM_WATERMARK_ENABLED env var)"
    )
    watermark_key: Optional[List[int]] = Field(
        default=None,
        description="Custom watermark key (5 integers). If not provided, uses server's private key (CSM_WATERMARK_KEY env var)"
    )


class TTSResponse(BaseModel):
    """Response model for TTS generation"""
    audio_base64: str
    sample_rate: int
    duration_ms: float
    processing_time_ms: float
    watermarked: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    last_request_time: Optional[float]
    watermark_enabled_default: bool
    watermark_key_configured: bool


def load_model():
    """Load CSM model - called once on first request (warm start)"""
    global generator
    if generator is not None:
        return generator
    
    print("Loading CSM-1B model...")
    start_time = time.time()
    
    # Import here to avoid loading at module import time
    from generator import load_csm_1b
    
    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("WARNING: CUDA not available, using CPU (will be slow)")
    
    generator = load_csm_1b(device)
    
    print(f"Model loaded in {time.time() - start_time:.2f}s on {device}")
    return generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Log configuration on startup
    print(f"Watermark enabled by default: {ENV_WATERMARK_ENABLED}")
    print(f"Private watermark key configured: {ENV_WATERMARK_KEY is not None}")
    yield
    pass


# Create FastAPI app
app = FastAPI(
    title="CSM TTS API",
    description="Text-to-Speech API using Sesame CSM-1B model with optional watermarking",
    version="1.1.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - use for keepalive pings"""
    global generator, last_request_time
    
    device = "unknown"
    if generator is not None:
        device = str(generator.device)
    elif torch.cuda.is_available():
        device = "cuda (not loaded)"
    else:
        device = "cpu (not loaded)"
    
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
        device=device,
        last_request_time=last_request_time,
        watermark_enabled_default=ENV_WATERMARK_ENABLED,
        watermark_key_configured=ENV_WATERMARK_KEY is not None
    )


@app.post("/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text using CSM model.
    
    First request will load the model (cold start ~30-60s).
    Subsequent requests use warm model (~1-3s).
    
    Watermark options:
    - watermark: true/false to enable/disable (default: server config)
    - watermark_key: custom 5-integer key (default: server's private key)
    """
    global generator, last_request_time
    
    start_time = time.time()
    last_request_time = start_time
    
    # Resolve watermark settings
    enable_watermark = request.watermark if request.watermark is not None else ENV_WATERMARK_ENABLED
    watermark_key = request.watermark_key if request.watermark_key is not None else ENV_WATERMARK_KEY
    
    try:
        # Load model if not already loaded (warm start)
        gen = load_model()
        
        # Generate audio with watermark options
        audio_tensor = gen.generate(
            text=request.text,
            speaker=request.speaker_id,
            context=[],  # No context for single utterance
            max_audio_length_ms=request.max_audio_length_ms,
            temperature=request.temperature,
            topk=request.topk,
            enable_watermark=enable_watermark,
            watermark_key=watermark_key,
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(
            buffer, 
            audio_tensor.unsqueeze(0).cpu(), 
            gen.sample_rate,
            format="wav"
        )
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Calculate duration
        duration_ms = (len(audio_tensor) / gen.sample_rate) * 1000
        processing_time_ms = (time.time() - start_time) * 1000
        
        return TTSResponse(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            sample_rate=gen.sample_rate,
            duration_ms=duration_ms,
            processing_time_ms=processing_time_ms,
            watermarked=enable_watermark
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/wav")
async def generate_speech_wav(request: TTSRequest):
    """
    Generate speech and return raw WAV audio directly.
    Useful for streaming to audio players.
    """
    global generator, last_request_time
    
    last_request_time = time.time()
    
    # Resolve watermark settings
    enable_watermark = request.watermark if request.watermark is not None else ENV_WATERMARK_ENABLED
    watermark_key = request.watermark_key if request.watermark_key is not None else ENV_WATERMARK_KEY
    
    try:
        gen = load_model()
        
        audio_tensor = gen.generate(
            text=request.text,
            speaker=request.speaker_id,
            context=[],
            max_audio_length_ms=request.max_audio_length_ms,
            temperature=request.temperature,
            topk=request.topk,
            enable_watermark=enable_watermark,
            watermark_key=watermark_key,
        )
        
        buffer = io.BytesIO()
        torchaudio.save(
            buffer, 
            audio_tensor.unsqueeze(0).cpu(), 
            gen.sample_rate,
            format="wav"
        )
        buffer.seek(0)
        
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Watermarked": str(enable_watermark).lower()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "CSM TTS API",
        "version": "1.1.0",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Generate speech (returns base64)",
            "/generate/wav": "Generate speech (returns WAV file)"
        },
        "watermark": {
            "enabled_default": ENV_WATERMARK_ENABLED,
            "private_key_configured": ENV_WATERMARK_KEY is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
