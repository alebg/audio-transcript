import json
import os
from typing import Literal

import whisper
from whisper import Whisper

from .models import DiarizedSegment, Err, Ok, SpeechSegmentWithAudio, SpeechSegmentWithTranscript

WhisperModelName = Literal["tiny", "base", "small", "medium", "large"]


def load_whisper_model(model_name: WhisperModelName = "base") -> Ok[Whisper] | Err:
    try:
        model = whisper.load_model(model_name)
        return Ok(value=model)
    except Exception as e:
        return Err(message=f"Failed to load {model_name} model: {e}")


def transcribe_audio(audio_filepath: str, model: Whisper) -> Ok[str] | Err:
    try:
        result = model.transcribe(audio=audio_filepath, language="fr")
        text: str = result["text"]
        return Ok(value=text.strip())
    except Exception as e:
        return Err(message=f"Failed to transcribe {audio_filepath}: {e}")


def transcribe_splitted_audio(
    speech_segments: list[SpeechSegmentWithAudio],
    model: Whisper,
) -> Ok[list[SpeechSegmentWithTranscript]] | Err:
    try:
        result: list[SpeechSegmentWithTranscript] = []
        for segment in speech_segments:
            tr = transcribe_audio(audio_filepath=segment.segment_audio_file, model=model)
            if isinstance(tr, Err):
                return tr
            result.append(SpeechSegmentWithTranscript(**segment.model_dump(), transcript=tr.value))
        return Ok(value=result)
    except Exception as e:
        return Err(message=f"Failed to transcribe segments: {e}")


def diarized_transcripts_to_json(
    speech_segments: list[SpeechSegmentWithTranscript],
    output_folder: str = "diarized_transcripts",
) -> Ok[str] | Err:
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

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


def main(
    speech_segments: list[SpeechSegmentWithAudio],
    model_name: WhisperModelName = "base",
    debug: bool = False,
) -> Ok[str] | Err:
    model_result = load_whisper_model(model_name=model_name)
    if isinstance(model_result, Err):
        return model_result

    tsa = transcribe_splitted_audio(speech_segments=speech_segments, model=model_result.value)
    if isinstance(tsa, Err):
        return tsa

    json_result = diarized_transcripts_to_json(speech_segments=tsa.value)
    if isinstance(json_result, Err):
        return json_result

    if debug:
        print(f"\ndiarized_transcripts.py:\n{json_result}\n")

    return json_result
