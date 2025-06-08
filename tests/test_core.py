"""Tests for core interfaces and data models."""

from pathlib import Path
from datetime import datetime

from video_asr_summary.core import (
    VideoInfo,
    AudioData,
    TranscriptionResult,
    SummaryResult,
    PipelineResult,
)


class TestDataModels:
    """Test the core data models."""
    
    def test_video_info_creation(self):
        """Test VideoInfo dataclass creation."""
        video_info = VideoInfo(
            file_path=Path("/test/video.mp4"),
            duration_seconds=120.5,
            frame_rate=30.0,
            width=1920,
            height=1080,
            file_size_bytes=50000000
        )
        
        assert video_info.file_path == Path("/test/video.mp4")
        assert video_info.duration_seconds == 120.5
        assert video_info.frame_rate == 30.0
        assert video_info.width == 1920
        assert video_info.height == 1080
        assert video_info.file_size_bytes == 50000000
    
    def test_audio_data_creation(self):
        """Test AudioData dataclass creation."""
        audio_data = AudioData(
            file_path=Path("/test/audio.wav"),
            duration_seconds=120.5,
            sample_rate=44100,
            channels=2,
            format="wav"
        )
        
        assert audio_data.file_path == Path("/test/audio.wav")
        assert audio_data.duration_seconds == 120.5
        assert audio_data.sample_rate == 44100
        assert audio_data.channels == 2
        assert audio_data.format == "wav"
    
    def test_transcription_result_creation(self):
        """Test TranscriptionResult dataclass creation."""
        transcription = TranscriptionResult(
            text="This is a test transcription.",
            confidence=0.95,
            segments=[{"start": 0.0, "end": 5.0, "text": "This is a test transcription."}],
            language="en",
            processing_time_seconds=2.5
        )
        
        assert transcription.text == "This is a test transcription."
        assert transcription.confidence == 0.95
        assert len(transcription.segments) == 1
        assert transcription.language == "en"
        assert transcription.processing_time_seconds == 2.5
    
    def test_summary_result_creation(self):
        """Test SummaryResult dataclass creation."""
        summary = SummaryResult(
            summary="This is a test summary.",
            key_points=["Point 1", "Point 2"],
            confidence=0.85,
            word_count=5,
            processing_time_seconds=1.2
        )
        
        assert summary.summary == "This is a test summary."
        assert summary.key_points == ["Point 1", "Point 2"]
        assert summary.confidence == 0.85
        assert summary.word_count == 5
        assert summary.processing_time_seconds == 1.2
    
    def test_pipeline_result_creation(self):
        """Test PipelineResult dataclass creation."""
        video_info = VideoInfo(
            file_path=Path("/test/video.mp4"),
            duration_seconds=120.5,
            frame_rate=30.0,
            width=1920,
            height=1080,
            file_size_bytes=50000000
        )
        
        audio_data = AudioData(
            file_path=Path("/test/audio.wav"),
            duration_seconds=120.5,
            sample_rate=44100,
            channels=2,
            format="wav"
        )
        
        transcription = TranscriptionResult(
            text="Test transcription",
            confidence=0.95,
            segments=[],
        )
        
        summary = SummaryResult(
            summary="Test summary",
            key_points=["Point 1"],
            confidence=0.85,
            word_count=2
        )
        
        timestamp = datetime.now()
        
        pipeline_result = PipelineResult(
            video_info=video_info,
            audio_data=audio_data,
            transcription=transcription,
            summary=summary,
            total_processing_time_seconds=10.5,
            timestamp=timestamp
        )
        
        assert pipeline_result.video_info == video_info
        assert pipeline_result.audio_data == audio_data
        assert pipeline_result.transcription == transcription
        assert pipeline_result.summary == summary
        assert pipeline_result.total_processing_time_seconds == 10.5
        assert pipeline_result.timestamp == timestamp
