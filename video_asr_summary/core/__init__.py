"""Core interfaces and data models for the video ASR summary pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from video_asr_summary.analysis import AnalysisResult


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
class SpeakerSegment:
    """A segment of audio attributed to a specific speaker."""

    start: float  # Start time in seconds
    end: float  # End time in seconds
    speaker: str  # Speaker identifier (e.g., "SPEAKER_00", "SPEAKER_01")
    confidence: float = 1.0  # Confidence score for speaker assignment


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""

    segments: list[SpeakerSegment]
    num_speakers: int
    processing_time_seconds: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Result of speech recognition."""

    text: str
    confidence: float
    segments: list[Dict[str, Any]]
    language: Optional[str] = None
    processing_time_seconds: Optional[float] = None


@dataclass
class EnhancedTranscriptionResult:
    """Transcription result enhanced with speaker information."""

    transcription: TranscriptionResult
    diarization: DiarizationResult
    # Each dict contains: 'start', 'end', 'text', 'speaker', 'confidence'
    # where 'speaker' is Optional[str] and 'confidence' is float (0.0-1.0)
    speaker_attributed_segments: list[Dict[str, Any]]  # Segments with speaker info
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
    analysis: Optional["AnalysisResult"]  # Content analysis result
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


class SpeakerDiarizationProcessor(ABC):
    """Abstract base class for speaker diarization."""

    @abstractmethod
    def diarize(
        self, audio_path: Path, num_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """Perform speaker diarization on audio."""
        pass


class ASRDiarizationIntegrator(ABC):
    """Abstract base class for integrating ASR and diarization results."""

    @abstractmethod
    def integrate(
        self, transcription: TranscriptionResult, diarization: DiarizationResult
    ) -> EnhancedTranscriptionResult:
        """Integrate transcription and diarization results."""
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
