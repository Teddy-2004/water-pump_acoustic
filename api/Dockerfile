# Dockerfile -- Acoustic Pump Diagnostics API
FROM python:3.11-slim

# librosa/soundfile need libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY data/ data/

ENV MODEL_PATH=models/pump_cnn_v1.h5
ENV METRICS_PATH=models/latest_metrics.json

EXPOSE 8000

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
