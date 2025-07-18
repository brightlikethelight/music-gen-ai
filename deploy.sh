#!/bin/bash
# MusicGen Deployment Script - Based on ACTUAL WORKING SOLUTIONS

set -e

echo "🚀 MusicGen Production Deployment"
echo "================================="
echo ""

# Option 1: Use Pre-built Working Docker Image (FASTEST)
deploy_prebuilt() {
    echo "📦 Option 1: Using Pre-built Docker Image (ashleykza/tts-webui)"
    echo "This image includes MusicGen + AudioGen + other TTS tools"
    echo ""
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker Desktop first."
        exit 1
    fi
    
    # Pull the working image
    echo "⬇️  Pulling pre-built image..."
    docker pull ashleykza/tts-webui:latest
    
    # Create local directories
    mkdir -p workspace outputs models
    
    # Run the container
    echo "🏃 Starting MusicGen container..."
    docker run -d \
        --name musicgen-prod \
        --gpus all \
        -v $(pwd)/workspace:/workspace \
        -v $(pwd)/outputs:/outputs \
        -v $(pwd)/models:/models \
        -p 3000:3001 \
        -p 8888:8888 \
        ashleykza/tts-webui:latest
    
    echo "✅ MusicGen is running!"
    echo "   Web UI: http://localhost:3000"
    echo "   Jupyter: http://localhost:8888"
}

# Option 2: Build Our Custom Image (CUSTOMIZABLE)
deploy_custom() {
    echo "🔨 Option 2: Building Custom Docker Image"
    echo ""
    
    # Create optimized Dockerfile
    cat > Dockerfile.working << 'EOF'
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies in correct order
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio>=2.0.0,<2.1.2 \
    numpy>=1.24.0,<2.0.0 \
    scipy==1.11.4

# Install transformers and audiocraft
RUN pip install --no-cache-dir \
    transformers>=4.31.0 \
    audiocraft>=1.1.0

# Install API dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    prometheus-client>=0.17.0 \
    structlog>=23.0.0

# Copy source code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/outputs /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "musicgen.api.rest.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Build the image
    echo "🏗️  Building Docker image..."
    docker build -f Dockerfile.working -t musicgen-custom:latest .
    
    # Run the container
    echo "🏃 Starting custom MusicGen container..."
    docker run -d \
        --name musicgen-custom \
        -v $(pwd)/outputs:/app/outputs \
        -v $(pwd)/models:/app/models \
        -p 8000:8000 \
        -e MODEL_NAME=facebook/musicgen-small \
        -e LOG_LEVEL=INFO \
        musicgen-custom:latest
    
    echo "✅ Custom MusicGen API is running!"
    echo "   API: http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
}

# Option 3: Local Python 3.10 Environment (NO DOCKER)
deploy_local() {
    echo "🐍 Option 3: Local Python 3.10 Environment"
    echo ""
    
    # Check if pyenv is installed
    if ! command -v pyenv &> /dev/null; then
        echo "📦 Installing pyenv..."
        curl https://pyenv.run | bash
        
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
        
        source ~/.zshrc
    fi
    
    # Install Python 3.10
    echo "🐍 Installing Python 3.10.14..."
    pyenv install 3.10.14
    pyenv local 3.10.14
    
    # Create virtual environment
    echo "🌍 Creating virtual environment..."
    python -m venv venv-musicgen
    source venv-musicgen/bin/activate
    
    # Install dependencies
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install torch==2.1.0 torchaudio>=2.0.0,<2.1.2
    pip install numpy>=1.24.0,<2.0.0 scipy==1.11.4
    pip install transformers>=4.31.0 audiocraft>=1.1.0
    pip install fastapi uvicorn[standard] python-multipart
    
    # Install our package
    pip install -e .
    
    echo "✅ Local environment ready!"
    echo "   Activate: source venv-musicgen/bin/activate"
    echo "   Run API: uvicorn musicgen.api.rest.app:app --reload"
}

# Option 4: Use Cloud Service (EASIEST)
deploy_cloud() {
    echo "☁️  Option 4: Cloud Services"
    echo ""
    echo "Ready-to-use MusicGen APIs:"
    echo ""
    echo "1. Replicate:"
    echo "   https://replicate.com/meta/musicgen"
    echo "   - Pay per use"
    echo "   - No setup required"
    echo "   - API access"
    echo ""
    echo "2. Hugging Face Spaces:"
    echo "   https://huggingface.co/spaces/facebook/MusicGen"
    echo "   - Free tier available"
    echo "   - Web interface"
    echo "   - Can duplicate for your own use"
    echo ""
    echo "3. RunPod:"
    echo "   - Deploy ashleykza/tts-webui template"
    echo "   - GPU instances"
    echo "   - Pay per hour"
}

# Test the deployment
test_deployment() {
    echo "🧪 Testing MusicGen API..."
    echo ""
    
    # Wait for server to start
    sleep 10
    
    # Test health endpoint
    echo "Testing health endpoint..."
    curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Health check failed"
    
    echo ""
    echo "Testing music generation..."
    
    # Create test request
    cat > test_request.json << 'EOF'
{
    "prompt": "happy acoustic guitar melody",
    "duration": 5.0,
    "model": "facebook/musicgen-small"
}
EOF
    
    # Send generation request
    RESPONSE=$(curl -s -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d @test_request.json)
    
    echo "Response: $RESPONSE"
    
    # Extract job ID
    JOB_ID=$(echo $RESPONSE | python -c "import json,sys; print(json.load(sys.stdin)['job_id'])" 2>/dev/null || echo "")
    
    if [ -n "$JOB_ID" ]; then
        echo "Job ID: $JOB_ID"
        echo ""
        echo "Checking status..."
        
        # Poll for completion
        for i in {1..30}; do
            STATUS=$(curl -s http://localhost:8000/status/$JOB_ID | python -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['status']} - {d.get('progress', 0)*100:.0f}%\")" 2>/dev/null || echo "error")
            echo "  $STATUS"
            
            if [[ $STATUS == *"completed"* ]]; then
                echo "✅ Generation completed!"
                break
            elif [[ $STATUS == *"failed"* ]]; then
                echo "❌ Generation failed!"
                break
            fi
            
            sleep 2
        done
    fi
    
    rm -f test_request.json
}

# Main menu
echo "Choose deployment option:"
echo "1) Pre-built Docker Image (Recommended)"
echo "2) Custom Docker Build"
echo "3) Local Python 3.10 Environment"
echo "4) Cloud Services (No setup)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        deploy_prebuilt
        test_deployment
        ;;
    2)
        deploy_custom
        test_deployment
        ;;
    3)
        deploy_local
        ;;
    4)
        deploy_cloud
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Next steps:"
echo "- For production: Set up nginx reverse proxy"
echo "- For scaling: Deploy to Kubernetes"
echo "- For monitoring: Set up Prometheus + Grafana"