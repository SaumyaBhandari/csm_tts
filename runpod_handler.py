"""
RunPod Serverless Handler for CSM TTS

This handler is designed for RunPod's serverless GPU platform.
- Auto-scales from 0 to N workers based on demand
- Model stays loaded in memory between requests (warm start)
- Automatic shutdown after idle timeout

Deploy:
1. Build Docker image with this handler
2. Push to container registry
3. Create RunPod serverless endpoint
"""
import os
import io
import base64
import time

import runpod
import torch
import torchaudio

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Global model instance (persists across requests for warm start)
generator = None


def load_model():
    """Load CSM model once, reuse across requests"""
    global generator
    if generator is not None:
        return generator
    
    print("Loading CSM-1B model...")
    start_time = time.time()
    
    from generator import load_csm_1b
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_csm_1b(device)
    
    print(f"Model loaded in {time.time() - start_time:.2f}s on {device}")
    return generator


def handler(job):
    """
    RunPod handler function.
    
    Input:
    {
        "input": {
            "text": "Hello world",
            "speaker_id": 0,
            "max_audio_length_ms": 30000,
            "temperature": 0.9,
            "topk": 50
        }
    }
    
    Output:
    {
        "audio_base64": "...",
        "sample_rate": 24000,
        "duration_ms": 1234.5,
        "processing_time_ms": 2345.6
    }
    """
    job_input = job.get("input", {})
    
    # Extract parameters with defaults
    text = job_input.get("text", "")
    speaker_id = job_input.get("speaker_id", 0)
    max_audio_length_ms = job_input.get("max_audio_length_ms", 30000)
    temperature = job_input.get("temperature", 0.9)
    topk = job_input.get("topk", 50)
    
    if not text:
        return {"error": "No text provided"}
    
    start_time = time.time()
    
    try:
        # Load model (warm start if already loaded)
        gen = load_model()
        
        # Generate audio
        audio_tensor = gen.generate(
            text=text,
            speaker=speaker_id,
            context=[],
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
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
        
        # Calculate metrics
        duration_ms = (len(audio_tensor) / gen.sample_rate) * 1000
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "sample_rate": gen.sample_rate,
            "duration_ms": duration_ms,
            "processing_time_ms": processing_time_ms
        }
        
    except Exception as e:
        return {"error": str(e)}


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
