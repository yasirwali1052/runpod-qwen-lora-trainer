# Use RunPod's official PyTorch image for maximum compatibility
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install git for Hugging Face transfers
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir unsloth

# Copy your local code (api_server.py, etc.) into the image
COPY . .

# Speed up the download on the RunPod side
ENV HF_HUB_ENABLE_HF_TRANSFER="1"
ENV HF_ENDPOINT="https://hf-mirror.com"

EXPOSE 8000

# Start the API
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]