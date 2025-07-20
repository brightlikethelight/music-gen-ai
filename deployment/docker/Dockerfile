# Multi-stage build for production MusicGen deployment
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .[deployment]

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r musicgen && useradd -r -g musicgen musicgen

# Set working directory
WORKDIR /app

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /app/src/ src/

# Set ownership
RUN chown -R musicgen:musicgen /app

# Switch to non-root user
USER musicgen

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "musicgen.api.rest.app:app", "--host", "0.0.0.0", "--port", "8000"]