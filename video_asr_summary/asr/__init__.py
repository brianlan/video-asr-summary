"""ASR (Automatic Speech Recognition) processors for the video ASR summary pipeline."""

from .whisper_processor import WhisperProcessor
from .funasr_processor import FunASRProcessor

__all__ = ["WhisperProcessor", "FunASRProcessor"]
