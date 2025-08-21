# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA-enabled PyTorch (match to CUDA version)
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# App deps
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt

# Copy code
COPY . /app

EXPOSE 8080
# 1–2 workers is often optimal for single-GPU inference; adjust after benchmarking
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]
