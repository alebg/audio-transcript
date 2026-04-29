import os
from pydub import AudioSegment

from .models import Err, Ok

rttm_columns = [
    "type",
    "file_id",
    "channel_id",
    "start_time",
    "duration",
    "orthographic_transcription",
    "speaker_type",
    "speaker_id",
    "confidence_score",
    "signal_lookahead_time",
]

# Map extensions to valid ffmpeg container formats
FORMAT_MAP = {
    "m4a": "mp4",  # ffmpeg uses mp4 container for m4a files
    "aac": "adts",  # raw AAC stream
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "opus": "opus",
}


def rename_file(old_filepath: str, new_filepath: str) -> Ok[str] | Err:
    try:
        os.rename(old_filepath, new_filepath)
        return Ok(value=new_filepath)
    except Exception as e:
        return Err(message=f"Failed to rename file: {e}")


def append_silence_segment(
    filepath: str,
    silence_miliseconds_duration: int = 3000,
) -> Ok[str] | Err:
    backup_filepath = f"{filepath}.bak"

    rename_result = rename_file(old_filepath=filepath, new_filepath=backup_filepath)
    if isinstance(rename_result, Err):
        return Err(message=f"Failed to create backup for {filepath}: {rename_result.message}")

    try:
        spacer = AudioSegment.silent(duration=silence_miliseconds_duration)
        audio = AudioSegment.from_file(backup_filepath)
        audio = spacer.append(audio, crossfade=0)

        original_ext = os.path.splitext(filepath)[1][1:].lower()
        export_format = FORMAT_MAP.get(original_ext)
        if not export_format:
            raise ValueError(f"Unsupported export format for extension: .{original_ext}")

        audio.export(filepath, format=export_format)
        return Ok(value=filepath)

    except Exception as e:
        return Err(message=f"Failed to append silence segment to {filepath}: {e}")


def convert_to_flac(filepath: str) -> Ok[str] | Err:
    if filepath.endswith(".flac"):
        return Ok(value=filepath)

    try:
        audio = AudioSegment.from_file(filepath)
        flac_filepath = f"{os.path.splitext(filepath)[0]}.flac"
        audio.export(flac_filepath, format="flac")
        return Ok(value=flac_filepath)
    except Exception as e:
        return Err(message=f"Failed to convert {filepath} to .flac format: {e}")
