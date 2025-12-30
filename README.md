# CSM TTS - Streaming API

Text-to-Speech API using Sesame CSM-1B model with real-time streaming support.

## Architecture

```
LLM Response → text chunks → CSM TTS Server → audio chunks → Browser
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/SaumyaBhandari/csm_tts.git
cd csm_tts
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login

# Run server
python api_server.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Single TTS (returns base64) |
| `/generate/wav` | POST | Single TTS (returns WAV) |
| `/stream` | WebSocket | **Real-time streaming TTS** |
| `/stream/chunks` | POST | HTTP streaming for batch |
| `/health` | GET | Health check |

## Streaming with WebSocket

Connect to `/stream` for real-time TTS as your LLM generates text:

```python
import websockets
import json
import asyncio

async def stream_tts():
    async with websockets.connect("ws://localhost:8000/stream") as ws:
        # Send text chunks as they arrive from LLM
        for chunk in ["Hello,", "how are you?", "I'm doing great!"]:
            await ws.send(json.dumps({"text": chunk}))
            
            # Receive audio immediately
            response = json.loads(await ws.recv())
            audio_base64 = response["audio_base64"]
            # Stream this audio to browser...
        
        # Close connection
        await ws.send(json.dumps({"done": True}))

asyncio.run(stream_tts())
```

## Environment Variables

```bash
# .env
CSM_WATERMARK_KEY=123,456,789,101,202  # Your private key
CSM_WATERMARK_ENABLED=true             # Default watermark state
HF_TOKEN=your_token                     # HuggingFace access
```

## Watermarking

Watermark embeds an imperceptible signature in audio for AI-generated content identification.

```json
// Disable watermark for this request
{"text": "Hello", "watermark": false}

// Use custom key
{"text": "Hello", "watermark_key": [1,2,3,4,5]}
```

## Docker Deployment

```bash
# Build
docker build -t csm-tts .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -e CSM_WATERMARK_KEY=1,2,3,4,5 \
  csm-tts
```

## Performance

| Metric | Cold Start | Warm Start |
|--------|------------|------------|
| Model Load | ~30-60s | 0s |
| Per Chunk | ~2-5s | ~2-5s |

## License

Apache-2.0 (see LICENSE)
