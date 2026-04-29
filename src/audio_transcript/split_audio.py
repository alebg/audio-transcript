import os
from pydub import AudioSegment

from .models import Err, Ok, SpeechSegment, SpeechSegmentWithAudio
from .utils import FORMAT_MAP


def split_audio(
    audio_filepath: str,
    speech_segments: list[SpeechSegment],
) -> Ok[list[SpeechSegmentWithAudio]] | Err:
    try:
        audio = AudioSegment.from_file(audio_filepath)
        base = os.path.splitext(audio_filepath)[0]
        ext = os.path.splitext(audio_filepath)[1][1:]

        export_format = FORMAT_MAP.get(ext)
        if not export_format:
            raise ValueError(f"Unsupported export format: {ext}")

        result: list[SpeechSegmentWithAudio] = []
        for i, segment in enumerate(speech_segments):
            output_file = f"{base}_part_{i}.{ext}"
            audio[int(segment.start_time * 1000) : int(segment.end_time * 1000)].export(
                output_file, format=export_format
            )
            result.append(SpeechSegmentWithAudio(**segment.model_dump(), segment_audio_file=output_file))

        return Ok(value=result)

    except Exception as e:
        return Err(message=f"Failed to split {audio_filepath}: {e}")


def splitted_audio_cleanup(speech_segments: list[SpeechSegmentWithAudio]) -> Ok[None] | Err:
    try:
        for segment in speech_segments:
            if os.path.isfile(segment.segment_audio_file):
                os.remove(segment.segment_audio_file)
        return Ok(value=None)
    except Exception as e:
        return Err(message=f"Failed to clean up segment audio files: {e}")


def main(
    audio_filepath: str,
    speech_segments: list[SpeechSegment],
    debug: bool = False,
) -> Ok[list[SpeechSegmentWithAudio]] | Err:
    result = split_audio(audio_filepath=audio_filepath, speech_segments=speech_segments)
    if debug:
        print(f"\nsplit_audio.py:\n{result}\n")
    return result


def cleanup(speech_segments: list[SpeechSegmentWithAudio], debug: bool = False) -> Ok[None] | Err:
    result = splitted_audio_cleanup(speech_segments=speech_segments)
    if debug:
        print(f"\nsplit_audio cleanup:\n{result}\n")
    return result
