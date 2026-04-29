#!/bin/bash
set -euo pipefail
HF_TOKEN=$(cat /run/secrets/hf_token)
git lfs install
git clone "https://user:${HF_TOKEN}@huggingface.co/pyannote/speaker-diarization-community-1" /model
rm -rf /model/.git
