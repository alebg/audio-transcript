"""
CLI entry point for the audio-transcript pipeline.

Subcommands
-----------
from-url    Download a YouTube video and produce a diarized transcript.
from-audio  Run the pipeline on an existing local audio file.

Examples
--------
    audio-transcript from-url https://youtu.be/xxx my_title --model small
    audio-transcript from-audio /path/to/audio.opus --model base
"""

import argparse
import os
import shutil
import sys

from . import diarized_transcripts, process_rttm_data, speaker_diarization, split_audio
from .diarized_transcripts import WhisperModelName
from .models import Err, Ok, WHISPER_LANGUAGES, WhisperLanguage, is_whisper_language
from .utils import FORMAT_MAP


def _validate_inputs(audio_filepath: str) -> Ok[None] | Err:
    """Fast checks that run before any heavy model work."""
    if not os.path.exists(audio_filepath):
        return Err(message=f"Audio file not found: '{audio_filepath}'")

    ext = os.path.splitext(audio_filepath)[1][1:].lower()
    if ext not in FORMAT_MAP:
        return Err(message=f"Unsupported audio format '.{ext}'. Supported: {sorted(FORMAT_MAP)}")

    if shutil.which("ffmpeg") is None:
        return Err(message="ffmpeg not found — install it with: sudo apt-get install ffmpeg")

    token_check = speaker_diarization.load_hf_access_token()
    if isinstance(token_check, Err):
        return token_check

    for directory in ("rttm_files", "data"):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            return Err(message=f"Cannot create output directory '{directory}': {e}")

    return Ok(value=None)


def _run_from_audio(
    audio_filepath: str, whisper_model: WhisperModelName, language: WhisperLanguage | None, debug: bool
) -> Ok[str] | Err:
    preflight = _validate_inputs(audio_filepath)
    if isinstance(preflight, Err):
        return preflight

    diarization = speaker_diarization.main(audio_filepath=audio_filepath, debug=debug)
    if isinstance(diarization, Err):
        return diarization

    rttm = process_rttm_data.main(rttm_filepath=diarization.value, debug=debug)
    if isinstance(rttm, Err):
        return rttm

    split = split_audio.main(audio_filepath=audio_filepath, speech_segments=rttm.value, debug=debug)
    if isinstance(split, Err):
        return split

    transcripts = diarized_transcripts.main(
        speech_segments=split.value, model_name=whisper_model, language=language, debug=debug
    )
    if isinstance(transcripts, Err):
        return transcripts

    cleanup = split_audio.cleanup(speech_segments=split.value)
    if isinstance(cleanup, Err):
        return cleanup

    return Ok(value=transcripts.value)


def _run_from_url(
    yt_url: str, file_title: str, whisper_model: WhisperModelName, language: WhisperLanguage | None, debug: bool
) -> Ok[str] | Err:
    from . import yt_audio_downloader

    download = yt_audio_downloader.main(yt_url=yt_url, file_title=file_title, debug=debug)
    if isinstance(download, Err):
        return download

    return _run_from_audio(
        audio_filepath=download.value,
        whisper_model=whisper_model,
        language=language,
        debug=debug,
    )


def _print_result(result: Ok[str] | Err) -> None:
    if isinstance(result, Ok):
        print(f"Success! Output: {result.value}")
    else:
        print(f"Error: {result.message}")
        sys.exit(1)


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="audio-transcript",
        description="YouTube video or audio file → speaker-diarized transcript",
    )
    parser.add_argument("--debug", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    url_parser = sub.add_parser("from-url", help="Download from YouTube and transcribe")
    url_parser.add_argument("url", help="YouTube video URL")
    url_parser.add_argument("title", help="Output file base name (no extension)")
    url_parser.add_argument(
        "--model", default="base", dest="whisper_model", choices=["tiny", "base", "small", "medium", "large"]
    )
    url_parser.add_argument(
        "--language",
        default=None,
        choices=sorted(WHISPER_LANGUAGES),
        metavar="LANG",
        help="ISO language code for Whisper (e.g. fr, en, de). Omit to auto-detect.",
    )

    audio_parser = sub.add_parser("from-audio", help="Transcribe an existing audio file")
    audio_parser.add_argument("audio_path", help="Path to the audio file")
    audio_parser.add_argument(
        "--model", default="base", dest="whisper_model", choices=["tiny", "base", "small", "medium", "large"]
    )
    audio_parser.add_argument(
        "--language",
        default=None,
        choices=sorted(WHISPER_LANGUAGES),
        metavar="LANG",
        help="ISO language code for Whisper (e.g. fr, en, de). Omit to auto-detect.",
    )

    args = parser.parse_args()

    raw_language = args.language
    language: WhisperLanguage | None = (
        raw_language if raw_language is not None and is_whisper_language(raw_language) else None
    )

    if args.command == "from-url":
        result = _run_from_url(
            yt_url=args.url,
            file_title=args.title,
            whisper_model=args.whisper_model,
            language=language,
            debug=args.debug,
        )
    else:
        result = _run_from_audio(
            audio_filepath=args.audio_path,
            whisper_model=args.whisper_model,
            language=language,
            debug=args.debug,
        )

    _print_result(result)
