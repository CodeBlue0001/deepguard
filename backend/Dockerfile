FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV, PyTorch audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create model weights dir
RUN mkdir -p model/weights

EXPOSE 8000

# Use gunicorn + uvicorn workers in production
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
