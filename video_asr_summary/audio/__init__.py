"""Audio processing components."""

from .extractor import (
    AudioExtractor,
    AudioExtractorFactory,
    FFmpegAudioExtractor,
    extract_audio_for_speech_recognition,
)

__all__ = [
    "AudioExtractor",
    "FFmpegAudioExtractor",
    "AudioExtractorFactory",
    "extract_audio_for_speech_recognition",
]
