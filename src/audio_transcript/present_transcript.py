import json

from .models import DiarizedSegment


def read_json(filepath: str) -> list[DiarizedSegment]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [DiarizedSegment.model_validate(item) for item in data]


def present_transcript(data: list[DiarizedSegment]) -> str:
    return "\n".join(f"{d.speaker_id}: {d.transcription}" for d in data)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print a diarized transcript JSON in readable form")
    parser.add_argument("-i", "--input", required=True, help="Path to the diarized transcript JSON file")
    args = parser.parse_args()

    print(present_transcript(read_json(args.input)))
