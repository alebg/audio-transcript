from typing import Any, Dict, List
import pandas as pd

from .utils import rttm_columns


def process_rttm(rttm_filepath: str) -> Dict[str, Any]:
    """Parses an RTTM file into a list of speech_segment dicts."""
    try:
        df = pd.read_csv(rttm_filepath, sep=" ", header=None, names=rttm_columns)
        df = df.fillna("<NA>")
        df["end_time"] = pd.to_numeric(df["start_time"] + df["duration"], errors='coerce').round(3)
        records: List[Dict[str, Any]] = df[["start_time", "end_time", "duration", "speaker_id"]].to_dict(orient="records")
        return {"status": True, "message": f"Successfully processed {rttm_filepath}", "data": records}
    except Exception as e:
        return {"status": False, "message": f"Failed to process rttm file: {e}"}


def filter_low_duration_speech_segments(
    speech_segments: List[Dict[str, Any]],
    min_duration: float = 0.3,
) -> Dict[str, Any]:
    try:
        filtered = [s for s in speech_segments if s['duration'] >= min_duration]
        return {
            "status": True,
            "message": f"Filtered segments shorter than {min_duration}s",
            "data": filtered,
        }
    except Exception as e:
        return {"status": False, "message": f"Failed to filter speech segments: {e}"}


def main(rttm_filepath: str, debug: bool = False) -> Dict[str, Any]:
    pr = process_rttm(rttm_filepath=rttm_filepath)
    if not pr["status"]:
        return pr

    fls = filter_low_duration_speech_segments(speech_segments=pr["data"])
    if not fls["status"]:
        return fls

    response: Dict[str, Any] = {
        "status": True,
        "message": f"Successfully processed {rttm_filepath} and filtered short segments",
        "data": fls["data"],
    }

    if debug:
        print(f"\nprocess_rttm_data.py:\n{response}\n")

    return response
