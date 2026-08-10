# Multi-stage build for production image
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -q -r requirements.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY models/ models/ 2>/dev/null || true

# Create uploads directory
RUN mkdir -p uploads results

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "-m", "src.api.app"]
