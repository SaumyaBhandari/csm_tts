"""
CSM TTS API Server

Standalone TTS server - receives text, returns audio.
Designed for serverless GPU deployment.

Endpoints:
    POST /generate     - Returns base64 audio
    POST /generate/wav - Returns WAV file directly
    GET  /health       - Health check

Environment Variables:
    CSM_WATERMARK_KEY: Private watermark key (comma-separated integers)
    CSM_WATERMARK_ENABLED: Enable watermark by default (true/false)
"""
import os
import io
import base64
import time
from typing import Optional, List

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

os.environ["NO_TORCH_COMPILE"] = "1"


# ============================================================================
# Configuration
# ============================================================================

def get_watermark_key() -> Optional[List[int]]:
    key_str = os.environ.get("CSM_WATERMARK_KEY", "")
    if not key_str:
        return None
    try:
        return [int(x.strip()) for x in key_str.split(",")]
    except ValueError:
        return None

ENV_WATERMARK_KEY = get_watermark_key()
ENV_WATERMARK_ENABLED = os.environ.get("CSM_WATERMARK_ENABLED", "true").lower() == "true"

generator = None


# ============================================================================
# Model
# ============================================================================

def load_model():
    global generator
    if generator is not None:
        return generator
    
    print("Loading CSM-1B model...")
    start = time.time()
    
    from generator import load_csm_1b
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_csm_1b(device)
    
    print(f"Loaded in {time.time() - start:.1f}s on {device}")
    return generator


# ============================================================================
# Request/Response Models
# ============================================================================

class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0
    max_audio_length_ms: int = 30000
    watermark: Optional[bool] = None
    watermark_key: Optional[List[int]] = None


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_ms: float
    processing_time_ms: float
    watermarked: bool


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="CSM TTS API",
    description="Text-to-Speech using Sesame CSM-1B",
    version="1.0.0"
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "device": str(generator.device) if generator else "not loaded"
    }


@app.post("/generate", response_model=TTSResponse)
async def generate(req: TTSRequest):
    """Generate speech from text, returns base64 audio"""
    start = time.time()
    
    enable_wm = req.watermark if req.watermark is not None else ENV_WATERMARK_ENABLED
    wm_key = req.watermark_key or ENV_WATERMARK_KEY
    
    try:
        gen = load_model()
        
        audio = gen.generate(
            text=req.text,
            speaker=req.speaker_id,
            context=[],
            max_audio_length_ms=req.max_audio_length_ms,
            temperature=0.9,
            topk=50,
            enable_watermark=enable_wm,
            watermark_key=wm_key,
        )
        
        buf = io.BytesIO()
        torchaudio.save(buf, audio.unsqueeze(0).cpu(), gen.sample_rate, format="wav")
        buf.seek(0)
        audio_bytes = buf.read()
        
        return TTSResponse(
            audio_base64=base64.b64encode(audio_bytes).decode(),
            sample_rate=gen.sample_rate,
            duration_ms=(len(audio) / gen.sample_rate) * 1000,
            processing_time_ms=(time.time() - start) * 1000,
            watermarked=enable_wm
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate/wav")
async def generate_wav(req: TTSRequest):
    """Generate speech from text, returns WAV file"""
    enable_wm = req.watermark if req.watermark is not None else ENV_WATERMARK_ENABLED
    wm_key = req.watermark_key or ENV_WATERMARK_KEY
    
    try:
        gen = load_model()
        
        audio = gen.generate(
            text=req.text,
            speaker=req.speaker_id,
            context=[],
            max_audio_length_ms=req.max_audio_length_ms,
            temperature=0.9,
            topk=50,
            enable_watermark=enable_wm,
            watermark_key=wm_key,
        )
        
        buf = io.BytesIO()
        torchaudio.save(buf, audio.unsqueeze(0).cpu(), gen.sample_rate, format="wav")
        buf.seek(0)
        
        return Response(content=buf.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/")
async def root():
    return {
        "name": "CSM TTS API",
        "endpoints": {
            "/generate": "POST - Returns base64 audio",
            "/generate/wav": "POST - Returns WAV file",
            "/health": "GET - Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
