# CSM TTS Server

Self-contained Text-to-Speech API using Sesame CSM-1B model.

## Project Structure

```
csm_tts/
├── main.py                 # Application entry point
├── app/
│   ├── config.py           # Configuration management
│   ├── api/
│   │   └── routes.py       # API endpoints
│   ├── core/
│   │   └── tts_service.py  # TTS business logic
│   └── models/
│       └── schemas.py      # Request/Response schemas
├── generator.py            # CSM model wrapper
├── models.py               # CSM model architecture
├── watermarking.py         # Audio watermarking
├── Dockerfile
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/generate` | Generate speech (base64) |
| POST | `/generate/wav` | Generate speech (WAV file) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login

# Run server
python main.py
```

## Usage

### Generate Speech (Base64)

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

Response:
```json
{
  "audio_base64": "UklGR...",
  "sample_rate": 24000,
  "duration_ms": 1500.0,
  "processing_time_ms": 2300.0,
  "watermarked": true
}
```

### Generate Speech (WAV File)

```bash
curl -X POST http://localhost:8000/generate/wav \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}' \
  --output speech.wav
```

### Full Request Options

```json
{
  "text": "Hello, world!",
  "speaker_id": 0,
  "max_audio_length_ms": 30000,
  "temperature": 0.9,
  "topk": 50,
  "watermark": true,
  "watermark_key": [1, 2, 3, 4, 5]
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `MODEL_DEVICE` | auto | Device (auto/cuda/cpu) |
| `CSM_WATERMARK_ENABLED` | true | Enable watermark |
| `CSM_WATERMARK_KEY` | - | Private watermark key |

## Docker Deployment

```bash
docker build -t csm-tts .
docker run --gpus all -p 8000:8000 csm-tts
```

## Requirements

- CUDA GPU (A10G, T4, or better)
- ~4GB GPU memory
- HuggingFace account (for model access)
