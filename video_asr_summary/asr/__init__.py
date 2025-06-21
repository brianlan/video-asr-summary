"""ASR (Automatic Speech Recognition) processors for the video ASR summary pipeline."""

from .whisper_processor import WhisperProcessor
from .funasr_specialized_processor import FunASRSpecializedProcessor

__all__ = ["WhisperProcessor", "FunASRSpecializedProcessor"]
