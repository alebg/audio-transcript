import hashlib
import json
import os
import shutil
from typing import Literal

import whisper
from whisper import Whisper

from .models import DiarizedSegment, Err, Ok, SpeechSegmentWithAudio, SpeechSegmentWithTranscript, WhisperLanguage

WhisperModelName = Literal["tiny", "base", "small", "medium", "large"]

_TRANSCRIPT_CACHE_DIR = ".cache/transcripts"


def _hash_segments(segments: list[SpeechSegmentWithAudio], model_name: str, language: WhisperLanguage | None) -> str:
    h = hashlib.sha256()
    for s in segments:
        h.update(f"{s.start_time:.3f}:{s.end_time:.3f}:{s.speaker_id}".encode())
    h.update(model_name.encode())
    h.update((language or "auto").encode())
    return h.hexdigest()


def load_whisper_model(model_name: WhisperModelName = "base") -> Ok[Whisper] | Err:
    try:
        model = whisper.load_model(model_name)
        return Ok(value=model)
    except Exception as e:
        return Err(message=f"Failed to load {model_name} model: {e}")


def transcribe_audio(audio_filepath: str, model: Whisper, language: WhisperLanguage | None) -> Ok[str] | Err:
    try:
        result = model.transcribe(audio=audio_filepath, language=language)
        text: str = result["text"]
        return Ok(value=text.strip())
    except Exception as e:
        return Err(message=f"Failed to transcribe {audio_filepath}: {e}")


def transcribe_splitted_audio(
    speech_segments: list[SpeechSegmentWithAudio],
    model: Whisper,
    language: WhisperLanguage | None,
) -> Ok[list[SpeechSegmentWithTranscript]] | Err:
    try:
        result: list[SpeechSegmentWithTranscript] = []
        for segment in speech_segments:
            tr = transcribe_audio(audio_filepath=segment.segment_audio_file, model=model, language=language)
            if isinstance(tr, Err):
                return tr
            result.append(SpeechSegmentWithTranscript(**segment.model_dump(), transcript=tr.value))
        return Ok(value=result)
    except Exception as e:
        return Err(message=f"Failed to transcribe segments: {e}")


def diarized_transcripts_to_json(
    speech_segments: list[SpeechSegmentWithTranscript],
    output_folder: str = "data",
) -> Ok[str] | Err:
    try:
        os.makedirs(output_folder, exist_ok=True)

        basename = os.path.splitext(os.path.basename(speech_segments[0].segment_audio_file))[0]
        basename = basename.replace("_part_0", "")
        output_file = f"{output_folder}/{basename}.json"

        records = [
            DiarizedSegment(
                start_time=s.start_time,
                end_time=s.end_time,
                duration=s.duration,
                speaker_id=s.speaker_id,
                transcription=s.transcript,
            ).model_dump()
            for s in speech_segments
        ]

        with open(output_file, "w") as f:
            json.dump(records, f, ensure_ascii=False)

        return Ok(value=output_file)

    except Exception as e:
        return Err(message=f"Failed to write diarized transcripts: {e}")


def _output_path_from_segments(
    segments: list[SpeechSegmentWithAudio],
    output_folder: str = "data",
) -> str:
    basename = os.path.splitext(os.path.basename(segments[0].segment_audio_file))[0]
    return f"{output_folder}/{basename.replace('_part_0', '')}.json"


def main(
    speech_segments: list[SpeechSegmentWithAudio],
    model_name: WhisperModelName = "base",
    language: WhisperLanguage | None = None,
    debug: bool = False,
) -> Ok[str] | Err:
    segment_hash = _hash_segments(speech_segments, model_name, language)
    cached = os.path.join(_TRANSCRIPT_CACHE_DIR, f"{segment_hash}.json")

    if os.path.exists(cached):
        output_file = _output_path_from_segments(speech_segments)
        os.makedirs("data", exist_ok=True)
        shutil.copy(cached, output_file)
        print(f"Transcript cache hit — skipping Whisper ({cached})")
        return Ok(value=output_file)

    model_result = load_whisper_model(model_name=model_name)
    if isinstance(model_result, Err):
        return model_result

    tsa = transcribe_splitted_audio(speech_segments=speech_segments, model=model_result.value, language=language)
    if isinstance(tsa, Err):
        return tsa

    json_result = diarized_transcripts_to_json(speech_segments=tsa.value)
    if isinstance(json_result, Err):
        return json_result

    os.makedirs(_TRANSCRIPT_CACHE_DIR, exist_ok=True)
    shutil.copy(json_result.value, cached)
    if debug:
        print(f"Transcript result cached to {cached}")

    if debug:
        print(f"\ndiarized_transcripts.py:\n{json_result}\n")

    return json_result
