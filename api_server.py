"""
CSM TTS API Server - Streaming Edition

FastAPI server with WebSocket support for real-time streaming TTS.
As text chunks arrive from your LLM, they're converted to audio and streamed back.

Flow:
    LLM → text chunks → CSM TTS → audio chunks → Browser

Environment Variables:
    CSM_WATERMARK_KEY: Private watermark key (comma-separated integers)
    CSM_WATERMARK_ENABLED: Enable watermark by default (true/false)
"""
import os
import io
import base64
import time
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"


# ============================================================================
# Configuration
# ============================================================================

def get_watermark_key_from_env() -> Optional[List[int]]:
    key_str = os.environ.get("CSM_WATERMARK_KEY", "")
    if not key_str:
        return None
    try:
        return [int(x.strip()) for x in key_str.split(",")]
    except ValueError:
        return None

ENV_WATERMARK_KEY = get_watermark_key_from_env()
ENV_WATERMARK_ENABLED = os.environ.get("CSM_WATERMARK_ENABLED", "true").lower() == "true"

# Global model
generator = None


# ============================================================================
# Model Loading
# ============================================================================

def load_model():
    """Load CSM model once (warm start)"""
    global generator
    if generator is not None:
        return generator
    
    print("Loading CSM-1B model...")
    start = time.time()
    
    from generator import load_csm_1b
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_csm_1b(device)
    
    print(f"Model loaded in {time.time() - start:.2f}s on {device}")
    return generator


def generate_audio(
    text: str,
    speaker_id: int = 0,
    max_audio_length_ms: int = 30000,
    enable_watermark: bool = True,
    watermark_key: Optional[List[int]] = None
) -> tuple[bytes, int]:
    """Generate audio bytes from text"""
    gen = load_model()
    
    audio_tensor = gen.generate(
        text=text,
        speaker=speaker_id,
        context=[],
        max_audio_length_ms=max_audio_length_ms,
        temperature=0.9,
        topk=50,
        enable_watermark=enable_watermark,
        watermark_key=watermark_key or ENV_WATERMARK_KEY,
    )
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, audio_tensor.unsqueeze(0).cpu(), gen.sample_rate, format="wav")
    buffer.seek(0)
    
    return buffer.read(), gen.sample_rate


# ============================================================================
# Request/Response Models
# ============================================================================

class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0
    max_audio_length_ms: int = 30000
    watermark: bool = True
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Watermark default: {ENV_WATERMARK_ENABLED}, key configured: {ENV_WATERMARK_KEY is not None}")
    yield

app = FastAPI(
    title="CSM TTS Streaming API",
    description="Real-time streaming TTS using Sesame CSM-1B",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "device": str(generator.device) if generator else "not loaded"
    }


@app.post("/generate", response_model=TTSResponse)
async def generate_speech(req: TTSRequest):
    """Generate speech from text (single request)"""
    start = time.time()
    
    try:
        audio_bytes, sample_rate = generate_audio(
            text=req.text,
            speaker_id=req.speaker_id,
            max_audio_length_ms=req.max_audio_length_ms,
            enable_watermark=req.watermark,
            watermark_key=req.watermark_key
        )
        
        # Calculate duration from WAV header (sample count)
        gen = load_model()
        
        return TTSResponse(
            audio_base64=base64.b64encode(audio_bytes).decode(),
            sample_rate=sample_rate,
            duration_ms=len(audio_bytes) / sample_rate * 1000 / 4,  # Approximate
            processing_time_ms=(time.time() - start) * 1000,
            watermarked=req.watermark
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/generate/wav")
async def generate_wav(req: TTSRequest):
    """Generate speech and return raw WAV file"""
    try:
        audio_bytes, _ = generate_audio(
            text=req.text,
            speaker_id=req.speaker_id,
            max_audio_length_ms=req.max_audio_length_ms,
            enable_watermark=req.watermark,
            watermark_key=req.watermark_key
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================================
# WebSocket Streaming Endpoint
# ============================================================================

@app.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.
    
    Protocol:
    1. Client connects
    2. Client sends JSON: {"text": "chunk of text", "speaker_id": 0, "watermark": true}
    3. Server responds with: {"audio_base64": "...", "sample_rate": 24000}
    4. Repeat for each text chunk
    5. Client sends: {"done": true} to close
    
    Example (from main backend):
        async with websockets.connect("ws://csm-server/stream") as ws:
            for chunk in llm_response_chunks:
                await ws.send(json.dumps({"text": chunk}))
                response = await ws.recv()
                audio_data = json.loads(response)["audio_base64"]
                # Stream audio to browser
    """
    await websocket.accept()
    print("Streaming client connected")
    
    # Ensure model is loaded on first connection
    load_model()
    
    try:
        while True:
            # Receive text chunk
            data = await websocket.receive_json()
            
            # Check for done signal
            if data.get("done"):
                await websocket.send_json({"status": "closed"})
                break
            
            text = data.get("text", "")
            if not text:
                await websocket.send_json({"error": "No text provided"})
                continue
            
            # Extract options
            speaker_id = data.get("speaker_id", 0)
            max_audio_length_ms = data.get("max_audio_length_ms", 30000)
            enable_watermark = data.get("watermark", ENV_WATERMARK_ENABLED)
            watermark_key = data.get("watermark_key", ENV_WATERMARK_KEY)
            
            start = time.time()
            
            try:
                # Generate audio for this chunk
                audio_bytes, sample_rate = generate_audio(
                    text=text,
                    speaker_id=speaker_id,
                    max_audio_length_ms=max_audio_length_ms,
                    enable_watermark=enable_watermark,
                    watermark_key=watermark_key
                )
                
                # Send audio back
                await websocket.send_json({
                    "audio_base64": base64.b64encode(audio_bytes).decode(),
                    "sample_rate": sample_rate,
                    "processing_time_ms": (time.time() - start) * 1000,
                    "text": text  # Echo back the text for syncing
                })
                
            except Exception as e:
                await websocket.send_json({"error": str(e), "text": text})
                
    except WebSocketDisconnect:
        print("Streaming client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


# ============================================================================
# HTTP Streaming Endpoint (Alternative to WebSocket)
# ============================================================================

@app.post("/stream/chunks")
async def stream_chunks(texts: List[str], speaker_id: int = 0, watermark: bool = True):
    """
    Stream audio for multiple text chunks.
    
    Request body: List of text strings
    Response: Streaming audio chunks as newline-delimited JSON
    
    Example:
        curl -X POST /stream/chunks 
             -H "Content-Type: application/json"
             -d '["Hello,", "how are you?", "I am fine."]'
    """
    async def generate_stream():
        for text in texts:
            if not text.strip():
                continue
            
            try:
                audio_bytes, sample_rate = generate_audio(
                    text=text,
                    speaker_id=speaker_id,
                    enable_watermark=watermark,
                    watermark_key=ENV_WATERMARK_KEY
                )
                
                chunk = {
                    "text": text,
                    "audio_base64": base64.b64encode(audio_bytes).decode(),
                    "sample_rate": sample_rate
                }
                yield f"{__import__('json').dumps(chunk)}\n"
                
            except Exception as e:
                yield f'{{"error": "{str(e)}", "text": "{text}"}}\n'
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"
    )


@app.get("/")
async def root():
    return {
        "name": "CSM TTS Streaming API",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Single TTS request (base64)",
            "/generate/wav": "Single TTS request (WAV file)",
            "/stream": "WebSocket streaming TTS",
            "/stream/chunks": "HTTP streaming for batch chunks"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
