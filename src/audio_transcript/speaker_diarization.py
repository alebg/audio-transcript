"""
Speaker diarization via pyannote/speaker-diarization-3.1.

Requires a Hugging Face access token with access granted at:
https://huggingface.co/pyannote/speaker-diarization-3.1

Set HUGGINGFACE_ACCESS_TOKEN in a .env file (see .env.template).
"""

import os
from typing import Any, Dict
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio
from dotenv import load_dotenv

from .utils import append_silence_segment


def load_hf_access_token() -> Dict[str, Any]:
    try:
        load_dotenv()
        token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    except Exception as e:
        return {"status": False, "message": f"Failed to load environment variables: {e}"}

    if token is None:
        return {
            "status": False,
            "message": "HUGGINGFACE_ACCESS_TOKEN not found. Add it to your .env file",
        }

    return {"status": True, "message": "Loaded HUGGINGFACE_ACCESS_TOKEN", "token": token}


def speaker_diarization(
    audio_file_path: str,
    hugging_face_access_token: str,
    preload_audio_in_memory: bool = False,
    output_folder: str = "rttm_files",
) -> Dict[str, Any]:
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hugging_face_access_token,
        )

        with ProgressHook() as hook:
            if preload_audio_in_memory:
                waveform, sample_rate = torchaudio.load(audio_file_path)
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
            else:
                diarization = pipeline(audio_file_path, hook=hook)

        basename = os.path.splitext(os.path.basename(audio_file_path))[0]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file = f"{output_folder}/{basename}.rttm"
        with open(output_file, "w") as rttm:
            diarization.write_rttm(rttm)

        return {
            "status": True,
            "message": f"Successfully ran speaker diarization on {audio_file_path}",
            "output_file": output_file,
        }

    except Exception as e:
        return {"status": False, "message": f"Failed to run speaker diarization on {audio_file_path}: {e}"}


def main(audio_filepath: str, debug: bool = False) -> Dict[str, Any]:
    token_response = load_hf_access_token()
    if not token_response["status"]:
        return token_response

    silence_response = append_silence_segment(filepath=audio_filepath)
    if not silence_response["status"]:
        return silence_response

    response = speaker_diarization(
        audio_file_path=silence_response["output_file"],
        hugging_face_access_token=token_response["token"],
        preload_audio_in_memory=True,
    )

    if debug:
        print(f"\nspeaker_diarization.py:\n{response}\n")

    return response
