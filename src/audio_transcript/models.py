from dataclasses import dataclass
from typing import Generic, Literal, TypeGuard, TypeVar, get_args

from pydantic import BaseModel

WhisperLanguage = Literal[
    'af',
    'am',
    'ar',
    'as',
    'az',
    'ba',
    'be',
    'bg',
    'bn',
    'bo',
    'br',
    'bs',
    'ca',
    'cs',
    'cy',
    'da',
    'de',
    'el',
    'en',
    'es',
    'et',
    'eu',
    'fa',
    'fi',
    'fo',
    'fr',
    'gl',
    'gu',
    'ha',
    'haw',
    'he',
    'hi',
    'hr',
    'ht',
    'hu',
    'hy',
    'id',
    'is',
    'it',
    'ja',
    'jw',
    'ka',
    'kk',
    'km',
    'kn',
    'ko',
    'la',
    'lb',
    'ln',
    'lo',
    'lt',
    'lv',
    'mg',
    'mi',
    'mk',
    'ml',
    'mn',
    'mr',
    'ms',
    'mt',
    'my',
    'ne',
    'nl',
    'nn',
    'no',
    'oc',
    'pa',
    'pl',
    'ps',
    'pt',
    'ro',
    'ru',
    'sa',
    'sd',
    'si',
    'sk',
    'sl',
    'sn',
    'so',
    'sq',
    'sr',
    'su',
    'sv',
    'sw',
    'ta',
    'te',
    'tg',
    'th',
    'tk',
    'tl',
    'tr',
    'tt',
    'uk',
    'ur',
    'uz',
    'vi',
    'yi',
    'yo',
    'yue',
    'zh',
]

WHISPER_LANGUAGES: frozenset[str] = frozenset(get_args(WhisperLanguage))


def is_whisper_language(s: str) -> TypeGuard[WhisperLanguage]:
    return s in WHISPER_LANGUAGES


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
