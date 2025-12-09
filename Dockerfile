# Use AMD ROCm PyTorch base
FROM rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.6.0

# Set working directory inside container
WORKDIR /workspace


RUN apt-get update && apt-get install -y \
    git \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
COPY pyproject.toml .
RUN python3.10 -m pip install --upgrade pip setuptools wheel
RUN python3.10 -m pip install -e .
RUN python3.10 -m pip install -e .[dev]
RUN apt-get update && apt-get install -y ffmpeg






