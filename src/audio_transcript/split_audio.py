import os
from typing import Any, Dict, List
from pydub import AudioSegment

from .utils import FORMAT_MAP


def split_audio(audio_filepath: str, speech_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Slices an audio file into per-segment files and appends segment_audio_file to each segment."""
    try:
        audio = AudioSegment.from_file(audio_filepath)
        base = os.path.splitext(audio_filepath)[0]
        ext = os.path.splitext(audio_filepath)[1][1:]

        export_format = FORMAT_MAP.get(ext)
        if not export_format:
            raise ValueError(f"Unsupported export format: {ext}")

        for i, segment in enumerate(speech_segments):
            start_ms = segment["start_time"] * 1000
            end_ms = segment["end_time"] * 1000
            output_file = f"{base}_part_{i}.{ext}"
            audio[start_ms:end_ms].export(output_file, format=export_format)
            segment["segment_audio_file"] = output_file

        return {"status": True, "message": f"Successfully split {audio_filepath}", "data": speech_segments}

    except Exception as e:
        return {"status": False, "message": f"Failed to split {audio_filepath}: {e}"}


def splitted_audio_cleanup(speech_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        for segment in speech_segments:
            if os.path.isfile(segment["segment_audio_file"]):
                os.remove(segment["segment_audio_file"])

        cleaned = [
            {"start_time": s["start_time"], "end_time": s["end_time"], "speaker_id": s["speaker_id"]}
            for s in speech_segments
        ]
        return {"status": True, "message": "Successfully cleaned up segment audio files", "data": cleaned}

    except Exception as e:
        return {"status": False, "message": f"Failed to clean up segment audio files: {e}"}


def main(audio_filepath: str, speech_segments: List[Dict[str, Any]], debug: bool = False) -> Dict[str, Any]:
    response = split_audio(audio_filepath=audio_filepath, speech_segments=speech_segments)
    if debug:
        print(f"\nsplit_audio.py:\n{response}\n")
    return response


def cleanup(speech_segments: List[Dict[str, Any]], debug: bool = False) -> Dict[str, Any]:
    response = splitted_audio_cleanup(speech_segments=speech_segments)
    if debug:
        print(f"\nsplit_audio cleanup:\n{response}\n")
    return response
