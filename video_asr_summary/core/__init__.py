"""Core interfaces and data models for the video ASR summary pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


@dataclass
class VideoInfo:
    """Information about a video file."""
    
    file_path: Path
    duration_seconds: float
    frame_rate: float
    width: int
    height: int
    file_size_bytes: int


@dataclass
class AudioData:
    """Extracted audio data from video."""
    
    file_path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str


@dataclass
class TranscriptionResult:
    """Result of speech recognition."""
    
    text: str
    confidence: float
    segments: list[Dict[str, Any]]
    language: Optional[str] = None
    processing_time_seconds: Optional[float] = None


@dataclass
class SummaryResult:
    """Result of text summarization."""
    
    summary: str
    key_points: list[str]
    confidence: float
    word_count: int
    processing_time_seconds: Optional[float] = None


@dataclass
class PipelineResult:
    """Complete pipeline processing result."""
    
    video_info: VideoInfo
    audio_data: AudioData
    transcription: TranscriptionResult
    summary: SummaryResult
    total_processing_time_seconds: float
    timestamp: datetime


class VideoProcessor(ABC):
    """Abstract base class for video processing."""
    
    @abstractmethod
    def extract_info(self, video_path: Path) -> VideoInfo:
        """Extract information from video file."""
        pass
    
    @abstractmethod
    def extract_audio(self, video_path: Path, output_path: Path) -> AudioData:
        """Extract audio from video file."""
        pass


class ASRProcessor(ABC):
    """Abstract base class for automatic speech recognition."""
    
    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio to text."""
        pass


class SummarizationProcessor(ABC):
    """Abstract base class for text summarization."""
    
    @abstractmethod
    def summarize(self, text: str) -> SummaryResult:
        """Generate summary from text."""
        pass


class Pipeline(ABC):
    """Abstract base class for the complete processing pipeline."""
    
    @abstractmethod
    def process(self, video_path: Path) -> PipelineResult:
        """Process video through the complete pipeline."""
        pass
