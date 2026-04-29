# audio-transcript

Converts a YouTube video (or any local audio file) into a speaker-diarized transcript.

**Pipeline stages:**
1. Download audio from YouTube (`yt-dlp`)
2. Identify speakers (`pyannote/speaker-diarization-3.1`)
3. Split audio by speaker segment (`pydub`)
4. Transcribe each segment (`openai-whisper`)
5. Save results as JSON

Output is a JSON array where each entry contains `start_time`, `end_time`, `duration`, `speaker_id`, and `transcription`.

## Requirements

- Python 3.11.9 (exactly — some ML libraries are not compatible with newer versions)
- FFmpeg (`sudo apt-get install ffmpeg`)
- A free Hugging Face account with access granted to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

`yt-dlp` must be installed from the nightly build (the PyPI release lags behind YouTube API changes):

```bash
pip install --force-reinstall https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz
```

Copy `.env.template` to `.env` and fill in your Hugging Face token:

```bash
cp .env.template .env
# then edit .env and set HUGGINGFACE_ACCESS_TOKEN
```

## Usage

### From a YouTube URL

```bash
audio-transcript from-url https://www.youtube.com/watch?v=<id> <output-title> [--model base]
```

### From a local audio file

```bash
audio-transcript from-audio /path/to/audio.opus [--model base]
```

### Print a saved transcript

```bash
present-transcript -i diarized_transcripts/<title>.json
```

### Whisper model sizes

| Model  | Notes                          |
|--------|--------------------------------|
| tiny   | Fastest, least accurate        |
| base   | Default                        |
| small  |                                |
| medium |                                |
| large  | Slowest, most accurate         |

## Output

Transcripts are saved to `diarized_transcripts/<title>.json`:

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

Intermediate files (`audio_files/`, `rttm_files/`) are created during the run. Segment audio slices are deleted automatically after transcription.

## Development

```bash
black .          # format
mypy .           # type check
pytest           # tests
```
