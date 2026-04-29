import json
import os
from typing import Any, Dict, List, Literal

import whisper
from whisper import Whisper


WhisperModelName = Literal["tiny", "base", "small", "medium", "large"]


def load_whisper_model(model_name: WhisperModelName = "base") -> Dict[str, Any]:
    try:
        model = whisper.load_model(model_name)
        return {"status": True, "message": f"Loaded {model_name} model", "model": model}
    except Exception as e:
        return {"status": False, "message": f"Failed to load {model_name} model: {e}"}


def transcribe_audio(audio_filepath: str, model: Whisper) -> Dict[str, Any]:
    try:
        result = model.transcribe(audio=audio_filepath, language="fr")
        return {"status": True, "message": f"Transcribed {audio_filepath}", "data": result}
    except Exception as e:
        return {"status": False, "message": f"Failed to transcribe {audio_filepath}: {e}"}


def transcribe_splitted_audio(speech_segments: List[Dict[str, Any]], model: Any) -> Dict[str, Any]:
    try:
        for segment in speech_segments:
            result = transcribe_audio(audio_filepath=segment["segment_audio_file"], model=model)
            if not result["status"]:
                return {"status": False, "message": result["message"]}
            segment["transcript"] = result["data"]["text"].strip()

        return {"status": True, "message": "Transcribed all segments", "data": speech_segments}

    except Exception as e:
        return {"status": False, "message": f"Failed to transcribe segments: {e}"}


def diarized_transcripts_to_json(
    speech_segments: List[Dict[str, Any]],
    output_folder: str = "diarized_transcripts",
) -> Dict[str, Any]:
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        basename = os.path.splitext(os.path.basename(speech_segments[0]["segment_audio_file"]))[0]
        basename = basename.replace("_part_0", "")
        output_file = f"{output_folder}/{basename}.json"

        records = [
            {
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "duration": s["duration"],
                "speaker_id": s["speaker_id"],
                "transcription": s["transcript"],
            }
            for s in speech_segments
        ]

        with open(output_file, "w") as f:
            json.dump(records, f, ensure_ascii=False)

        return {"status": True, "message": f"Wrote diarized transcripts to {output_file}", "output_file": output_file}

    except Exception as e:
        return {"status": False, "message": f"Failed to write diarized transcripts: {e}"}


def main(
    speech_segments: List[Dict[str, Any]],
    model_name: WhisperModelName = "base",
    debug: bool = False,
) -> Dict[str, Any]:
    model_response = load_whisper_model(model_name=model_name)
    if not model_response["status"]:
        return {"status": False, "message": f"Failed to load model: {model_response['message']}"}

    tsa = transcribe_splitted_audio(speech_segments=speech_segments, model=model_response["model"])
    if not tsa["status"]:
        return {"status": False, "message": f"Failed to transcribe segments: {tsa['message']}"}

    json_response = diarized_transcripts_to_json(speech_segments=tsa["data"])
    if not json_response["status"]:
        return {"status": False, "message": f"Failed to write transcripts: {json_response['message']}"}

    if debug:
        print(f"\ndiarized_transcripts.py:\n{json_response}\n")

    return json_response
