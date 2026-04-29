#!/usr/bin/env bash
# Runs the audio-transcript pipeline inside Docker.
# Pulls the latest image from GHCR; falls back to a local build prompt if unavailable.
# Output files are written with the current user's ownership (not root).
set -euo pipefail

IMAGE="ghcr.io/alebg/audio-transcript:latest"

# Load .env so HF_TOKEN (and other vars) are available to this script
if [ -f .env ]; then
    set -a && source .env && set +a
fi

_build_locally() {
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "Error: HF_TOKEN is required to build the image (the diarization model is gated)."
        echo "Set HF_TOKEN in your .env file and try again."
        exit 1
    fi
    echo "Building with HF_TOKEN from .env..."
    docker build --secret "id=hf_token,env=HF_TOKEN" -t "$IMAGE" .
}

# --- Image resolution ---
if docker pull "$IMAGE" 2>/dev/null; then
    : # up to date
elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Warning: could not pull latest image — using cached local version."
else
    echo "Image '$IMAGE' is not available remotely or locally."
    printf 'Build it locally now? [y/N] '
    read -r answer
    if [[ "${answer,,}" == "y" ]]; then
        _build_locally
    else
        echo "Aborted. Run 'docker build -t $IMAGE .' to build locally."
        exit 1
    fi
fi

# --- Ensure host-side output dirs exist (owned by current user) ---
mkdir -p data .cache/diarization .cache/transcripts

# --- Run ---
# --user:        container process runs as the current host UID/GID → output files
#                are owned by the host user, not root
# HOME=/app:     points whisper's model cache (~/.cache/whisper) at the mounted
#                .cache volume so it persists across runs
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/app \
    -e PYTHONUNBUFFERED=1 \
    -e MPLCONFIGDIR=/app/.cache/matplotlib \
    -e TORCHINDUCTOR_CACHE_DIR=/app/.cache/torch \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/.cache:/app/.cache" \
    "$IMAGE" \
    "$@"
