"""Main package initialization."""

from .core import (
    ASRProcessor,
    AudioData,
    Pipeline,
    PipelineResult,
    SummarizationProcessor,
    SummaryResult,
    TranscriptionResult,
    VideoInfo,
    VideoProcessor,
)

__version__ = "0.1.0"
__all__ = [
    "VideoInfo",
    "AudioData",
    "TranscriptionResult",
    "SummaryResult",
    "PipelineResult",
    "VideoProcessor",
    "ASRProcessor",
    "SummarizationProcessor",
    "Pipeline",
]
