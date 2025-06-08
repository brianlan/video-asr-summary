"""Main package initialization."""

from .core import (
    VideoInfo,
    AudioData,
    TranscriptionResult,
    SummaryResult,
    PipelineResult,
    VideoProcessor,
    ASRProcessor,
    SummarizationProcessor,
    Pipeline,
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
