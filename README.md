# CSM TTS Server

Standalone Text-to-Speech API using Sesame CSM-1B model.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Returns base64 audio |
| `/generate/wav` | POST | Returns WAV file |
| `/health` | GET | Health check |

## Usage

```bash
# Generate audio (base64)
curl -X POST http://server:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Generate audio (WAV file)
curl -X POST http://server:8000/generate/wav \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  --output speech.wav
```

## Request Body

```json
{
  "text": "Text to speak",
  "speaker_id": 0,
  "max_audio_length_ms": 30000,
  "watermark": true,
  "watermark_key": [1, 2, 3, 4, 5]
}
```

## Environment Variables

```bash
CSM_WATERMARK_KEY=1,2,3,4,5    # Private watermark key
CSM_WATERMARK_ENABLED=true     # Enable watermark by default
```

## Deployment

```bash
docker build -t csm-tts .
docker run --gpus all -p 8000:8000 csm-tts
```

## Requirements

- CUDA GPU (A10G, T4, or better)
- ~4GB GPU memory
