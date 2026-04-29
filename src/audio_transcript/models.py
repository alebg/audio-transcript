from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar('T')


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err:
    message: str


class SpeechSegment(BaseModel):
    start_time: float
    end_time: float
    duration: float
    speaker_id: str


class SpeechSegmentWithAudio(SpeechSegment):
    segment_audio_file: str


class SpeechSegmentWithTranscript(SpeechSegmentWithAudio):
    transcript: str


class DiarizedSegment(BaseModel):
    start_time: float
    end_time: float
    duration: float
    speaker_id: str
    transcription: str
