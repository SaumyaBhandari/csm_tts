# CSM TTS Serverless Deployment

This repo contains the CSM (Conversational Speech Model) TTS model, configured for serverless GPU deployment.

## Quick Start (Local)

```bash
# Clone and setup
git clone https://github.com/SaumyaBhandari/csm_tts.git
cd csm_tts
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for model access)
huggingface-cli login

# Run API server
python api_server.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check (use for keepalive) |
| `/generate` | POST | Generate speech (returns base64) |
| `/generate/wav` | POST | Generate speech (returns WAV file) |

### Generate Speech

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "speaker_id": 0}'
```

Response:
```json
{
  "audio_base64": "UklGR...",
  "sample_rate": 24000,
  "duration_ms": 1500.0,
  "processing_time_ms": 2300.0
}
```

## Serverless Deployment Options

### Option 1: RunPod Serverless (Recommended)

```bash
# Build and push Docker image
docker build -t your-registry/csm-tts:latest .
docker push your-registry/csm-tts:latest

# Deploy on RunPod with GPU
# Configure: Min workers=0, Max workers=3, GPU=A10G or better
```

### Option 2: AWS Lambda with GPU (via AWS Inferentia or SageMaker)

```bash
# Use SageMaker serverless inference endpoint
# See: https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html
```

### Option 3: Modal.com

```python
# See modal_deploy.py for Modal deployment
modal deploy modal_deploy.py
```

## Warm Start vs Cold Start

| Metric | Cold Start | Warm Start |
|--------|------------|------------|
| Model Load | ~30-60s | 0s (cached) |
| Generation | ~2-5s | ~2-5s |
| **Total** | ~35-65s | ~2-5s |

The server keeps the model in memory. First request loads the model (cold start), subsequent requests are fast (warm).

## Architecture

```
[Client] --> [API Server] --> [CSM Model] --> [Audio WAV]
                 |
                 v
           [HuggingFace Hub]
           (model weights)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | (from huggingface-cli login) |
| `NO_TORCH_COMPILE` | Disable Triton compilation | 1 |
| `PORT` | Server port | 8000 |
