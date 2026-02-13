# Base image with CUDA runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Avoid interactive prompts and set Python output
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python, pip, git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements file
COPY docker/requirements.txt .

# Upgrade pip and install torch + torchvision first, then other dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.0+cu121 \
        torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy project files
COPY docker/qwen_processor.py .
COPY docker/api_server.py .

# Create workspace folders
RUN mkdir -p /workspace/input /workspace/output

# Environment variables for Hugging Face cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TRANSFORMERS_CACHE=/workspace/.cache

# Expose API port
EXPOSE 8000

# Start the API server
CMD ["python", "-u", "api_server.py"]
