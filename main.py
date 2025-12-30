"""
CSM TTS API Server

Self-contained Text-to-Speech API using Sesame CSM-1B model.

Usage:
    python main.py
    # or
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    settings = get_settings()
    print("=" * 50)
    print("CSM TTS Server Starting")
    print("=" * 50)
    print(f"Watermark enabled: {settings.watermark_enabled}")
    print(f"Watermark key configured: {settings.watermark_key is not None}")
    print(f"Device: {settings.get_device()}")
    print("=" * 50)
    yield
    print("CSM TTS Server Shutting Down")


# Create FastAPI application
app = FastAPI(
    title="CSM TTS API",
    description="""
## Text-to-Speech API using Sesame CSM-1B

Convert text to natural-sounding speech.

### Features
- High-quality speech synthesis
- Multiple speaker voices
- Optional audio watermarking
- Base64 or WAV file output

### Usage

```python
import requests

response = requests.post(
    "http://server:8000/generate",
    json={"text": "Hello, world!"}
)
audio_base64 = response.json()["audio_base64"]
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/", tags=["System"])
async def root():
    """API information"""
    return {
        "name": "CSM TTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /generate": "Generate speech (base64)",
            "POST /generate/wav": "Generate speech (WAV file)",
            "GET /health": "Health check"
        }
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
