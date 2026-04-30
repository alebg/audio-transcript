import argparse

import pytest

from audio_transcript.format_transcript import (
    _ensure_sentence_end,
    _format_segment,
    _format_simple,
    _group_consecutive,
    _parse_speakers,
    _resolve_name,
    _to_hms,
    process_timestamps,
)
from audio_transcript.models import DiarizedSegment


def _seg(speaker_id: str, transcription: str, start: float = 0.0, end: float = 1.0) -> DiarizedSegment:
    return DiarizedSegment(
        start_time=start,
        end_time=end,
        duration=end - start,
        speaker_id=speaker_id,
        transcription=transcription,
    )


# --- _to_hms ---


@pytest.mark.parametrize(
    'seconds, expected',
    [
        (0.0, '00:00:00'),
        (59.0, '00:00:59'),
        (60.0, '00:01:00'),
        (180.9, '00:03:01'),  # rounds up
        (180.4, '00:03:00'),  # rounds down
        (3600.0, '01:00:00'),
        (3661.0, '01:01:01'),
        (86399.0, '23:59:59'),
    ],
)
def test_to_hms(seconds: float, expected: str) -> None:
    assert _to_hms(seconds) == expected


# --- process_timestamps ---


def test_process_timestamps_same_rounded() -> None:
    # 180.0 and 180.4 both round to 180 → same HH:MM:SS
    assert process_timestamps(180.0, 180.4) == '00:03:00'


def test_process_timestamps_different() -> None:
    assert process_timestamps(180.0, 181.0) == '00:03:00 – 00:03:01'


def test_process_timestamps_identical() -> None:
    assert process_timestamps(0.0, 0.0) == '00:00:00'


# --- _parse_speakers ---


def test_parse_speakers_single() -> None:
    assert _parse_speakers('0=John Smith') == {'SPEAKER_00': 'John Smith'}


def test_parse_speakers_multiple() -> None:
    result = _parse_speakers('0=John Smith,1=Mary Mueller')
    assert result == {'SPEAKER_00': 'John Smith', 'SPEAKER_01': 'Mary Mueller'}


def test_parse_speakers_non_sequential() -> None:
    assert _parse_speakers('2=Charlie') == {'SPEAKER_02': 'Charlie'}


def test_parse_speakers_strips_whitespace() -> None:
    assert _parse_speakers('0 = John Smith') == {'SPEAKER_00': 'John Smith'}


def test_parse_speakers_invalid_key_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match='integer'):
        _parse_speakers('abc=John')


def test_parse_speakers_missing_equals_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match='Invalid speaker entry'):
        _parse_speakers('0John')


def test_parse_speakers_empty_name_raises() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match='Invalid speaker entry'):
        _parse_speakers('0=')


# --- _resolve_name ---


def test_resolve_name_found() -> None:
    assert _resolve_name('SPEAKER_00', {'SPEAKER_00': 'John'}) == 'John'


def test_resolve_name_not_found() -> None:
    assert _resolve_name('SPEAKER_00', {}) == 'SPEAKER_00'


# --- _format_segment ---


def test_format_segment_with_name() -> None:
    seg = _seg('SPEAKER_00', 'Hello there', start=60.0, end=62.0)
    result = _format_segment(seg, {'SPEAKER_00': 'John'})
    assert result == 'John (00:01:00 – 00:01:02): "Hello there"'


def test_format_segment_without_name() -> None:
    seg = _seg('SPEAKER_01', 'Hi', start=0.0, end=0.0)
    result = _format_segment(seg, {})
    assert result == 'SPEAKER_01 (00:00:00): "Hi"'


# --- _format_simple ---


def test_format_simple_with_name() -> None:
    assert _format_simple('SPEAKER_00', 'Hello world', {'SPEAKER_00': 'John'}) == 'John: "Hello world"'


def test_format_simple_without_name() -> None:
    assert _format_simple('SPEAKER_00', 'Hello world', {}) == 'SPEAKER_00: "Hello world"'


# --- _ensure_sentence_end ---


@pytest.mark.parametrize(
    'text, expected',
    [
        ('Wechseln', 'Wechseln.'),  # ends with alpha → append dot
        ('Okay super.', 'Okay super.'),  # ends with dot → unchanged
        ('Really?', 'Really?'),  # ends with punctuation → unchanged
        ('Yes!', 'Yes!'),  # ends with punctuation → unchanged
        ('', ''),  # empty → unchanged
        ('Gut,', 'Gut,'),  # ends with comma → unchanged
    ],
)
def test_ensure_sentence_end(text: str, expected: str) -> None:
    assert _ensure_sentence_end(text) == expected


# --- _group_consecutive ---


def test_group_consecutive_merges_same_speaker() -> None:
    segments = [
        _seg('SPEAKER_00', 'Wechseln'),
        _seg('SPEAKER_00', 'Okay super.'),
    ]
    result = list(_group_consecutive(segments))
    assert result == [('SPEAKER_00', 'Wechseln. Okay super.')]


def test_group_consecutive_splits_on_interleave() -> None:
    segments = [
        _seg('SPEAKER_00', 'Hello'),
        _seg('SPEAKER_01', 'Hi'),
        _seg('SPEAKER_00', 'Back again'),
    ]
    result = list(_group_consecutive(segments))
    assert result == [
        ('SPEAKER_00', 'Hello.'),
        ('SPEAKER_01', 'Hi.'),
        ('SPEAKER_00', 'Back again.'),
    ]


def test_group_consecutive_empty() -> None:
    assert list(_group_consecutive([])) == []


def test_group_consecutive_single() -> None:
    result = list(_group_consecutive([_seg('SPEAKER_00', 'Only one')]))
    assert result == [('SPEAKER_00', 'Only one.')]
