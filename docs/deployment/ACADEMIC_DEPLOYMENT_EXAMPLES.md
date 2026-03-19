# 🚀 MusicGen Production Deployment Guide

## Python Version Recommendation

Python **3.10 or 3.11** is recommended for the smoothest experience. Python 3.12 is tested in CI and works with the current dependency set, but some ML ecosystem packages may have rough edges on 3.12.

## 📊 Deployment Options Comparison

| Option | Setup Time | Cost | Performance | Reliability | Customization |
|--------|------------|------|-------------|-------------|---------------|
| Pre-built Docker | 5 min | Free* | Good | ⭐⭐⭐⭐⭐ | Limited |
| Custom Docker | 30 min | Free* | Good | ⭐⭐⭐⭐ | Full |
| Local Python 3.10 | 20 min | Free | Good | ⭐⭐⭐ | Full |
| Cloud Services | 1 min | Pay-per-use | Excellent | ⭐⭐⭐⭐⭐ | Limited |

*Requires your own hardware

## 🏆 Option 1: Pre-built Docker Image (RECOMMENDED)

### Why This Works
- Uses Python 3.11.12 with verified dependencies
- Includes MusicGen, AudioGen, and Bark
- Tested by thousands of users
- Optimized for RunPod deployment

### Quick Start
```bash
# Pull the image
docker pull ashleykza/tts-webui:latest

# Run with GPU support
docker run -d \
    --name musicgen \
    --gpus all \
    -v $(pwd)/outputs:/outputs \
    -p 3000:3001 \
    ashleykza/tts-webui:latest

# Access at http://localhost:3000
```

### Pros
- ✅ Zero configuration
- ✅ GPU support included
- ✅ Web UI included
- ✅ Battle-tested

### Cons
- ❌ Large image size (20GB+)
- ❌ Includes extra tools you might not need
- ❌ Less control over API

## 🔧 Option 2: Custom Docker Build

### Optimized Dockerfile (Python 3.10)
```dockerfile
FROM python:3.10-slim

# Recommended: Python 3.10 or 3.11 (3.12 also works)
RUN apt-get update && apt-get install -y \
    git ffmpeg build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install in specific order to avoid conflicts
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio>=2.0.0,<2.1.2 \
    numpy>=1.24.0,<2.0.0 \
    scipy==1.11.4

RUN pip install --no-cache-dir \
    transformers>=4.31.0 \
    audiocraft>=1.1.0

# Add your API layer
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "musicgen.api.rest.app:app", "--host", "0.0.0.0"]
```

### Build and Run
```bash
docker build -t musicgen-api .
docker run -d -p 8000:8000 musicgen-api
```

## 🐍 Option 3: Local Python 3.10 Environment

### Using pyenv (macOS/Linux)
```bash
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.10
pyenv install 3.10.14
pyenv local 3.10.14

# Create virtual environment
python -m venv venv-musicgen
source venv-musicgen/bin/activate

# Install dependencies
pip install torch==2.1.0 torchaudio>=2.0.0,<2.1.2
pip install numpy>=1.24.0,<2.0.0 scipy==1.11.4
pip install transformers>=4.31.0 audiocraft>=1.1.0
```

## ☁️ Option 4: Cloud Services (Fastest)

### Replicate (Pay-per-use)
```python
import replicate

output = replicate.run(
    "meta/musicgen:7be0f12c54a8d033a0fbd14418c9af98962da9a86f5ff7811f9b3423a1f0b7d7",
    input={
        "prompt": "Edo25 major g melodies that sound triumphant and cinematic",
        "duration": 8
    }
)
```

### Hugging Face Inference API
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

response = requests.post(API_URL, headers=headers, json={
    "inputs": "happy rock music",
})
```

## 🏗️ Production Architecture

### Recommended Stack
```yaml
Load Balancer: nginx
├── API Layer: FastAPI (3 instances)
│   └── Queue: RabbitMQ/Celery
│       └── Workers: MusicGen (GPU nodes)
├── Storage: S3/MinIO
├── Cache: Redis
└── Monitoring: Prometheus + Grafana
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: musicgen-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: musicgen-api
  template:
    metadata:
      labels:
        app: musicgen-api
    spec:
      containers:
      - name: musicgen
        image: musicgen-api:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
```

## 💾 Hardware Requirements

### Model Memory Requirements
| Model | VRAM | System RAM | Generation Speed |
|-------|------|------------|------------------|
| Small (300M) | 4GB | 8GB | ~10s for 30s audio |
| Medium (1.5B) | 8GB | 16GB | ~30s for 30s audio |
| Large (3.3B) | 16GB | 32GB | ~60s for 30s audio |
| Melody | 8GB | 16GB | ~30s for 30s audio |

### Recommended Hardware
- **Minimum**: RTX 3060 (12GB) or T4 (16GB)
- **Optimal**: RTX 3090 (24GB) or A10G (24GB)
- **Production**: A100 (40GB/80GB)

## 🚦 Performance Optimization

### 1. Model Loading
```python
# Cache models in memory
model_cache = {}

async def get_model(model_name: str):
    if model_name not in model_cache:
        model_cache[model_name] = MusicGen.get_pretrained(model_name)
    return model_cache[model_name]
```

### 2. Batch Processing
```python
# Process multiple requests together
def batch_generate(prompts: List[str], model):
    with torch.cuda.amp.autocast():
        return model.generate(prompts, progress=True)
```

### 3. Memory Management
```python
# Clear cache after generation
torch.cuda.empty_cache()

# Use smaller precision
model = model.to(torch.float16)
```

## 🔍 Monitoring

### Key Metrics
- Generation latency (p50, p95, p99)
- Queue depth
- GPU utilization
- Memory usage
- Error rates

### Prometheus Queries
```promql
# Average generation time
rate(musicgen_generation_duration_seconds_sum[5m]) 
/ rate(musicgen_generation_duration_seconds_count[5m])

# GPU memory usage
nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes

# Request rate
rate(musicgen_generation_requests_total[1m])
```

## 🛡️ Security Considerations

### API Security
```python
# Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/generate")
@limiter.limit("10/minute")
async def generate_music(request: GenerationRequest):
    # Implementation
```

### Input Validation
```python
# Sanitize prompts
def sanitize_prompt(prompt: str) -> str:
    # Remove special characters
    prompt = re.sub(r'[^\w\s-]', '', prompt)
    # Limit length
    return prompt[:256]
```

## 🚨 Common Issues and Solutions

### Issue 1: CUDA Out of Memory
```python
# Solution: Reduce batch size or use smaller model
try:
    output = model.generate([prompt])
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    output = model.generate([prompt])
```

### Issue 2: Slow CPU Generation
```python
# Solution: Ensure CUDA is available
if torch.cuda.is_available():
    model = model.cuda()
else:
    warnings.warn("GPU not available, generation will be slow")
```

### Issue 3: Import Errors
```bash
# Solution: Install in correct order
pip uninstall torch torchaudio -y
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Scaling Strategies

### Horizontal Scaling
- Use message queue (RabbitMQ/Celery)
- Deploy multiple worker nodes
- Load balance API requests

### Vertical Scaling
- Use larger GPU instances
- Increase batch sizes
- Enable model parallelism

### Caching Strategy
- Cache generated audio for common prompts
- Use Redis for job status
- CDN for audio delivery

## 🎯 Production Checklist

- [ ] Use Python 3.10, 3.11, or 3.12
- [ ] Test with actual GPU
- [ ] Implement health checks
- [ ] Set up monitoring
- [ ] Configure auto-scaling
- [ ] Implement rate limiting
- [ ] Set up backup storage
- [ ] Configure SSL/TLS
- [ ] Document API endpoints
- [ ] Create runbooks

## 🆘 Quick Fixes

```bash
# Fix 1: Reset environment
rm -rf venv/
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Fix 2: Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Fix 3: Test minimal setup
python -c "from audiocraft.models import MusicGen; model = MusicGen.get_pretrained('facebook/musicgen-small')"
```

## 📞 Support Resources

- **AudioCraft GitHub**: https://github.com/facebookresearch/audiocraft
- **Hugging Face Forums**: https://discuss.huggingface.co/
- **RunPod Discord**: For deployment help
- **Stack Overflow**: Tag with `musicgen` or `audiocraft`

---

**Remember**: The key to successful MusicGen deployment is using the right Python version (3.10 or 3.11) and following the dependency installation order exactly. When in doubt, use pre-built Docker images that are known to work.