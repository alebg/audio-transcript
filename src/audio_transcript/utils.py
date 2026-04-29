import os
from typing import Any, Dict
from pydub import AudioSegment


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
    "m4a": "mp4",   # ffmpeg uses mp4 container for m4a files
    "aac": "adts",  # raw AAC stream
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "opus": "opus",
}


def rename_file(old_filepath: str, new_filepath: str) -> Dict[str, Any]:
    try:
        os.rename(old_filepath, new_filepath)
        return {"status": True, "message": "Successfully renamed file", "filepath": new_filepath}
    except Exception as e:
        return {"status": False, "message": f"Failed to rename file: {e}"}


def append_silence_segment(
    filepath: str,
    silence_miliseconds_duration: int = 3000,
) -> Dict[str, Any]:
    backup_filepath = f"{filepath}.bak"

    rename_response = rename_file(old_filepath=filepath, new_filepath=backup_filepath)
    if not rename_response['status']:
        return {"status": False, "message": f"Failed to create backup for {filepath}: {rename_response['message']}"}

    try:
        spacer = AudioSegment.silent(duration=silence_miliseconds_duration)
        audio = AudioSegment.from_file(backup_filepath)
        audio = spacer.append(audio, crossfade=0)

        original_ext = os.path.splitext(filepath)[1][1:].lower()
        export_format = FORMAT_MAP.get(original_ext)
        if not export_format:
            raise ValueError(f"Unsupported export format for extension: .{original_ext}")

        audio.export(filepath, format=export_format)
        return {"status": True, "message": f"Successfully appended silence segment to {filepath}", "output_file": filepath}

    except Exception as e:
        return {"status": False, "message": f"Failed to append silence segment to {filepath}: {e}"}


def convert_to_flac(filepath: str) -> Dict[str, Any]:
    if filepath.endswith(".flac"):
        return {"status": True, "message": f"{filepath} is already in .flac format. Skipping conversion"}

    try:
        audio = AudioSegment.from_file(filepath)
        flac_filepath = f"{os.path.splitext(filepath)[0]}.flac"
        audio.export(flac_filepath, format="flac")
        return {"status": True, "message": f"Successfully converted {filepath} to .flac format", "output_file": flac_filepath}
    except Exception as e:
        return {"status": False, "message": f"Failed to convert {filepath} to .flac format: {e}"}
