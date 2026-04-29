"""
Speaker diarization via a locally-cloned pyannote model.

The model must be cloned before running (once):

    git lfs install
    git clone https://huggingface.co/pyannote/speaker-diarization-community-1 /path/to/model

Then set DIARIZATION_MODEL_PATH in your .env file (see .env.template).
In the Docker image the model is baked in at /model and the env var is pre-set.
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


def load_model_path() -> Ok[str] | Err:
    load_dotenv()
    model_path = os.getenv("DIARIZATION_MODEL_PATH")
    if not model_path:
        return Err(message="DIARIZATION_MODEL_PATH is not set. Add it to your .env file.")
    if not os.path.isdir(model_path):
        return Err(message=f"DIARIZATION_MODEL_PATH '{model_path}' does not exist or is not a directory.")
    return Ok(value=model_path)


def speaker_diarization(
    audio_file_path: str,
    model_path: str,
    preload_audio_in_memory: bool = False,
    output_folder: str = "data/rttm_files",
) -> Ok[str] | Err:
    try:
        pipeline = Pipeline.from_pretrained(model_path)
        if not isinstance(pipeline, Pipeline):
            raise RuntimeError(f"Pipeline.from_pretrained('{model_path}') returned None")

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
    model_path_result = load_model_path()
    if isinstance(model_path_result, Err):
        return model_path_result

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
        model_path=model_path_result.value,
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
