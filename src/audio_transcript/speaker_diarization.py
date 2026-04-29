"""
Speaker diarization via pyannote/speaker-diarization-3.1.

Requires a Hugging Face access token with access granted at:
https://huggingface.co/pyannote/speaker-diarization-3.1

Set HUGGINGFACE_ACCESS_TOKEN in a .env file (see .env.template).
"""

import hashlib
import os
import shutil

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio
from dotenv import load_dotenv

from .models import Err, Ok
from .utils import append_silence_segment

_CACHE_DIR = ".cache/diarization"


def _hash_audio(filepath: str) -> str:
    # If silence was already prepended on a prior run, .bak holds the original content
    source = f"{filepath}.bak" if os.path.exists(f"{filepath}.bak") else filepath
    h = hashlib.sha256()
    with open(source, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_rttm(file_hash: str) -> str:
    return os.path.join(_CACHE_DIR, f"{file_hash}.rttm")


def load_hf_access_token() -> Ok[str] | Err:
    try:
        load_dotenv()
        token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    except Exception as e:
        return Err(message=f"Failed to load environment variables: {e}")

    if token is None:
        return Err(message="HUGGINGFACE_ACCESS_TOKEN not found. Add it to your .env file")

    return Ok(value=token)


def speaker_diarization(
    audio_file_path: str,
    hugging_face_access_token: str,
    preload_audio_in_memory: bool = False,
    output_folder: str = "rttm_files",
) -> Ok[str] | Err:
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hugging_face_access_token,
        )
        if not isinstance(pipeline, Pipeline):
            raise RuntimeError("Pipeline.from_pretrained returned None — check your HuggingFace token and model access")

        with ProgressHook() as hook:
            if preload_audio_in_memory:
                waveform, sample_rate = torchaudio.load(audio_file_path)
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
            else:
                diarization = pipeline(audio_file_path, hook=hook)

        basename = os.path.splitext(os.path.basename(audio_file_path))[0]
        os.makedirs(output_folder, exist_ok=True)
        output_file = f"{output_folder}/{basename}.rttm"

        # pyannote >=3.2 wraps the result in DiarizeOutput; older versions return Annotation directly
        annotation = diarization.speaker_diarization if hasattr(diarization, 'speaker_diarization') else diarization
        with open(output_file, "w") as rttm:
            annotation.write_rttm(rttm)

        return Ok(value=output_file)

    except Exception as e:
        return Err(message=f"Failed to run speaker diarization on {audio_file_path}: {e}")


def main(audio_filepath: str, debug: bool = False) -> Ok[str] | Err:
    token_result = load_hf_access_token()
    if isinstance(token_result, Err):
        return token_result

    file_hash = _hash_audio(audio_filepath)
    cached = _cached_rttm(file_hash)

    if os.path.exists(cached):
        print(f"Diarization cache hit — skipping embeddings ({cached})")
        return Ok(value=cached)

    silence_result = append_silence_segment(filepath=audio_filepath)
    if isinstance(silence_result, Err):
        return silence_result

    result = speaker_diarization(
        audio_file_path=silence_result.value,
        hugging_face_access_token=token_result.value,
        preload_audio_in_memory=True,
    )

    if isinstance(result, Ok):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        shutil.copy(result.value, cached)
        if debug:
            print(f"Diarization result cached to {cached}")

    if debug:
        print(f"\nspeaker_diarization.py:\n{result}\n")

    return result
