import json
from typing import Any, Dict, List, Union

JSONAtom = Dict[str, Any]
TJSON = Union[JSONAtom, List[JSONAtom]]


def read_json(filepath: str) -> TJSON:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def present_transcript(data: TJSON) -> str:
    if isinstance(data, list):
        return "\n".join(f"{d['speaker_id']}: {d['transcription']}" for d in data)
    return "\n".join(f"{k}: {v}" for k, v in data.items())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print a diarized transcript JSON in readable form")
    parser.add_argument("-i", "--input", required=True, help="Path to the diarized transcript JSON file")
    args = parser.parse_args()

    data = read_json(args.input)
    print(present_transcript(data))
