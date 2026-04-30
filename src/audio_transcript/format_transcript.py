import argparse
import logging
import sys
from pathlib import Path
from typing import Generator

from pydantic import TypeAdapter

from .models import DiarizedSegment

logger = logging.getLogger(__name__)

_adapter: TypeAdapter[list[DiarizedSegment]] = TypeAdapter(list[DiarizedSegment])


def _to_hms(seconds: float) -> str:
    total = round(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f'{h:02d}:{m:02d}:{s:02d}'


def process_timestamps(start_time: float, end_time: float) -> str:
    start = _to_hms(start_time)
    end = _to_hms(end_time)
    if start == end:
        return start
    return f'{start} – {end}'


def _format_segment(seg: DiarizedSegment) -> str:
    ts = process_timestamps(seg.start_time, seg.end_time)
    return f'{seg.speaker_id} ({ts}): "{seg.transcription}"'


def _read_segments(json_path: Path) -> Generator[DiarizedSegment, None, None]:
    segments = _adapter.validate_json(json_path.read_text())
    return (seg for seg in segments if seg.transcription)


def _write_lines(lines: Generator[str, None, None], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def _pipeline(json_path: Path, output_path: Path) -> None:
    segments = _read_segments(json_path)
    lines = (_format_segment(seg) for seg in segments)
    _write_lines(lines, output_path)
    logger.info('Written to %s', output_path)


def cli() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='Format a diarized transcript JSON to plain text.')
    parser.add_argument('input', type=Path, help='Path to the .json transcript file')
    args = parser.parse_args()

    json_path: Path = args.input
    if not json_path.exists():
        logger.error('File not found: %s', json_path)
        sys.exit(1)

    output_path = Path('data/formatted') / json_path.with_suffix('.txt').name
    _pipeline(json_path, output_path)


if __name__ == '__main__':
    cli()
