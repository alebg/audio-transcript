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
import pprint
import shutil
import sys
from typing import Any, Dict

from . import diarized_transcripts, process_rttm_data, speaker_diarization, split_audio
from .diarized_transcripts import WhisperModelName
from .utils import FORMAT_MAP


def _validate_inputs(audio_filepath: str) -> Dict[str, Any]:
    """Fast checks that run before any heavy model work."""
    if not os.path.exists(audio_filepath):
        return {"status": False, "message": f"Audio file not found: '{audio_filepath}'"}

    ext = os.path.splitext(audio_filepath)[1][1:].lower()
    if ext not in FORMAT_MAP:
        return {"status": False, "message": f"Unsupported audio format '.{ext}'. Supported: {sorted(FORMAT_MAP)}"}

    if shutil.which("ffmpeg") is None:
        return {"status": False, "message": "ffmpeg not found — install it with: sudo apt-get install ffmpeg"}

    token_check = speaker_diarization.load_hf_access_token()
    if not token_check["status"]:
        return token_check

    for directory in ("rttm_files", "diarized_transcripts"):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            return {"status": False, "message": f"Cannot create output directory '{directory}': {e}"}

    return {"status": True, "message": "All pre-flight checks passed"}


def _run_from_audio(audio_filepath: str, whisper_model: WhisperModelName, debug: bool) -> Dict[str, Any]:
    preflight = _validate_inputs(audio_filepath)
    if not preflight["status"]:
        return preflight

    diarization = speaker_diarization.main(audio_filepath=audio_filepath, debug=debug)
    if not diarization["status"]:
        return diarization

    rttm = process_rttm_data.main(rttm_filepath=diarization["output_file"], debug=debug)
    if not rttm["status"]:
        return rttm

    split = split_audio.main(audio_filepath=audio_filepath, speech_segments=rttm["data"], debug=debug)
    if not split["status"]:
        return split

    transcripts = diarized_transcripts.main(speech_segments=split["data"], model_name=whisper_model, debug=debug)
    if not transcripts["status"]:
        return transcripts

    cleanup = split_audio.cleanup(speech_segments=split["data"])
    if not cleanup["status"]:
        return cleanup

    return {
        "status": True,
        "message": "Pipeline completed successfully",
        "output_file": transcripts["output_file"],
    }


def _run_from_url(yt_url: str, file_title: str, whisper_model: WhisperModelName, debug: bool) -> Dict[str, Any]:
    from . import yt_audio_downloader

    download = yt_audio_downloader.main(yt_url=yt_url, file_title=file_title, debug=debug)
    if not download["status"]:
        return download

    return _run_from_audio(
        audio_filepath=download["output_file"],
        whisper_model=whisper_model,
        debug=debug,
    )


def _print_result(response: Dict[str, Any]) -> None:
    if response["status"]:
        print(f"Success: {pprint.pformat(response['message'])}")
        print(f"Output file: {response['output_file']}")
    else:
        print(f"Error: {pprint.pformat(response['message'])}")
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
    url_parser.add_argument("--model", default="base", dest="whisper_model",
                            choices=["tiny", "base", "small", "medium", "large"])

    audio_parser = sub.add_parser("from-audio", help="Transcribe an existing audio file")
    audio_parser.add_argument("audio_path", help="Path to the audio file")
    audio_parser.add_argument("--model", default="base", dest="whisper_model",
                              choices=["tiny", "base", "small", "medium", "large"])

    args = parser.parse_args()

    if args.command == "from-url":
        response = _run_from_url(
            yt_url=args.url,
            file_title=args.title,
            whisper_model=args.whisper_model,
            debug=args.debug,
        )
    else:
        response = _run_from_audio(
            audio_filepath=args.audio_path,
            whisper_model=args.whisper_model,
            debug=args.debug,
        )

    _print_result(response)
