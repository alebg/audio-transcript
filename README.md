# audio-transcript

Converts a YouTube video (or any local audio file) into a speaker-diarized transcript.

**Pipeline stages:**
1. Download audio from YouTube (`yt-dlp`)
2. Identify speakers (`pyannote/speaker-diarization-community-1`)
3. Split audio by speaker segment (`pydub`)
4. Transcribe each segment (`openai-whisper`)
5. Save results as JSON

Output is a JSON array where each entry contains `start_time`, `end_time`, `duration`, `speaker_id`, and `transcription`.

---

## Docker (recommended)

The Docker image has the diarization model baked in — no HuggingFace token or internet access needed at runtime.

### Run

```bash
./run.sh from-audio data/my-audio.m4a --language fr
./run.sh from-url https://www.youtube.com/watch?v=<id> my-title --language fr --model small
```

`run.sh` pulls the latest image from GHCR automatically. If the image is not available remotely or locally it offers to build it.

All output lands under `data/`:

| Path | Contents |
|---|---|
| `data/<title>.json` | Final transcript |
| `data/rttm_files/` | Speaker diarization (RTTM) |
| `data/audio_files/` | Downloaded YouTube audio (`from-url` only) |

Diarization and transcript results are cached in `.cache/` so re-runs on the same file skip the expensive steps.

### Build locally

```bash
docker build -t ghcr.io/alebg/audio-transcript:latest .
```

The image is pushed automatically by the GitHub Actions workflow (`.github/workflows/release.yml`) on every published release, tagged with the release version and `:latest`. Set the `HF_TOKEN` secret in **Settings → Secrets and variables → Actions** before triggering a release.

On first push: **GitHub → Packages → audio-transcript → Package settings → Visibility → Private**.

---

## Local (without Docker)

### Requirements

- Python 3.11.9 exactly — some ML libraries are not compatible with newer versions
- FFmpeg: `sudo apt-get install ffmpeg`
- The diarization model downloaded locally (once):

```bash
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('pyannote/speaker-diarization-community-1', local_dir='/path/to/model')"
```

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

`yt-dlp` must be the nightly build (PyPI lags behind YouTube API changes):

```bash
pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
```

Copy `.env.template` to `.env` and set the model path:

```bash
cp .env.template .env
# edit .env: DIARIZATION_MODEL_PATH=/path/to/model
```

### Usage

```bash
audio-transcript from-audio data/my-audio.m4a --language fr
audio-transcript from-url https://www.youtube.com/watch?v=<id> my-title --language fr
present-transcript -i data/<title>.json
```

### Whisper model sizes

| Model | Notes |
|---|---|
| tiny | Fastest, least accurate |
| base | Default |
| small | |
| medium | |
| large | Slowest, most accurate |

---

## Output format

```json
[
  {
    "start_time": 4.711,
    "end_time": 22.283,
    "duration": 17.572,
    "speaker_id": "SPEAKER_01",
    "transcription": "..."
  }
]
```

---

## Development

```bash
black .     # format
mypy src/   # type check
pytest      # tests
```
