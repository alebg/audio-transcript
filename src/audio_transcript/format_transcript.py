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


def _parse_speakers(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in raw.split(','):
        key, sep, name = pair.partition('=')
        if not sep or not name.strip():
            raise argparse.ArgumentTypeError(f'Invalid speaker entry {pair!r} — expected format: 0=Name')
        try:
            idx = int(key.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(f'Speaker key must be an integer, got {key!r}')
        result[f'SPEAKER_{idx:02d}'] = name.strip()
    return result


def _format_segment(seg: DiarizedSegment, speaker_names: dict[str, str]) -> str:
    ts = process_timestamps(seg.start_time, seg.end_time)
    name = speaker_names.get(seg.speaker_id, seg.speaker_id)
    return f'{name} ({ts}): "{seg.transcription}"'


def _write_lines(lines: Generator[str, None, None], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def _pipeline(json_path: Path, output_path: Path, speaker_names: dict[str, str]) -> None:
    all_segments = _adapter.validate_json(json_path.read_text())

    if speaker_names:
        actual_ids = frozenset(seg.speaker_id for seg in all_segments)
        for sid in sorted(frozenset(speaker_names) - actual_ids):
            logger.warning('Speaker "%s" provided but not found in transcript — ignored', sid)

    segments = (seg for seg in all_segments if seg.transcription)
    lines = (_format_segment(seg, speaker_names) for seg in segments)
    _write_lines(lines, output_path)
    logger.info('Written to %s', output_path)


def cli() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description='Format a diarized transcript JSON to plain text.')
    parser.add_argument('input', type=Path, help='Path to the .json transcript file')
    parser.add_argument(
        '-s',
        '--speakers',
        metavar='SPEAKERS',
        default='',
        help='Speaker name map, e.g. "0=John Smith,1=Mary Mueller"',
    )
    args = parser.parse_args()

    json_path: Path = args.input
    if not json_path.exists():
        logger.error('File not found: %s', json_path)
        sys.exit(1)

    speaker_names: dict[str, str] = _parse_speakers(args.speakers) if args.speakers else {}
    output_path = Path('data/formatted') / json_path.with_suffix('.txt').name
    _pipeline(json_path, output_path, speaker_names)


if __name__ == '__main__':
    cli()
