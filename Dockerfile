FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency-caching layer: heavy pip install runs only when pyproject.toml changes.
COPY pyproject.toml .
RUN mkdir -p src/audio_transcript && touch src/audio_transcript/__init__.py \
    && pip install --upgrade pip setuptools wheel \
    && pip install "." \
    && pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz \
    && rm -rf src/

# Download diarization model at build time.
# The model is gated — a HuggingFace token is always required.
# Token is mounted as a BuildKit secret so it never lands in a layer.
COPY scripts/download_model.sh /tmp/download_model.sh
RUN --mount=type=secret,id=hf_token \
    bash /tmp/download_model.sh && \
    chmod -R a+rX /model && \
    rm /tmp/download_model.sh

COPY src/ src/
RUN pip install --no-deps -e .

ENV DIARIZATION_MODEL_PATH=/model \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    TORCHINDUCTOR_CACHE_DIR=/app/.cache/torch

ENTRYPOINT ["audio-transcript"]
