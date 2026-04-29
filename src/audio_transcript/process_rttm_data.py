import pandas as pd

from .models import Err, Ok, SpeechSegment


def process_rttm(rttm_filepath: str) -> Ok[list[SpeechSegment]] | Err:
    try:
        from .utils import rttm_columns

        df = pd.read_csv(rttm_filepath, sep=" ", header=None, names=rttm_columns)
        df = df.fillna("<NA>")
        df["end_time"] = pd.to_numeric(df["start_time"] + df["duration"], errors='coerce').round(3)
        df = df[["start_time", "end_time", "duration", "speaker_id"]]

        records = [
            SpeechSegment(
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                duration=float(row["duration"]),
                speaker_id=str(row["speaker_id"]),
            )
            for _, row in df.iterrows()
        ]
        return Ok(value=records)
    except Exception as e:
        return Err(message=f"Failed to process rttm file: {e}")


def filter_low_duration_speech_segments(
    speech_segments: list[SpeechSegment],
    min_duration: float = 0.3,
) -> Ok[list[SpeechSegment]] | Err:
    try:
        filtered = [s for s in speech_segments if s.duration >= min_duration]
        return Ok(value=filtered)
    except Exception as e:
        return Err(message=f"Failed to filter speech segments: {e}")


def main(rttm_filepath: str, debug: bool = False) -> Ok[list[SpeechSegment]] | Err:
    pr = process_rttm(rttm_filepath=rttm_filepath)
    if isinstance(pr, Err):
        return pr

    fls = filter_low_duration_speech_segments(speech_segments=pr.value)
    if isinstance(fls, Err):
        return fls

    if debug:
        print(f"\nprocess_rttm_data.py: {len(fls.value)} segments after filtering\n")

    return fls
