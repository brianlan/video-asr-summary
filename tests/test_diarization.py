"""Tests for speaker diarization functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from video_asr_summary.core import (
    DiarizationResult,
    EnhancedTranscriptionResult,
    SpeakerSegment,
    TranscriptionResult,
)
from video_asr_summary.diarization.integrator import SegmentBasedIntegrator
from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor


class TestSpeakerSegment:
    """Test cases for SpeakerSegment data model."""

    def test_speaker_segment_creation(self):
        """Test creating a speaker segment."""
        segment = SpeakerSegment(
            start=1.0, end=3.5, speaker="SPEAKER_00", confidence=0.95
        )

        assert segment.start == 1.0
        assert segment.end == 3.5
        assert segment.speaker == "SPEAKER_00"
        assert segment.confidence == 0.95

    def test_speaker_segment_default_confidence(self):
        """Test default confidence value."""
        segment = SpeakerSegment(start=0.0, end=1.0, speaker="SPEAKER_01")

        assert segment.confidence == 1.0


class TestDiarizationResult:
    """Test cases for DiarizationResult data model."""

    def test_diarization_result_creation(self):
        """Test creating a diarization result."""
        segments = [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(2.5, 4.0, "SPEAKER_01"),
        ]

        result = DiarizationResult(
            segments=segments, num_speakers=2, processing_time_seconds=1.5
        )

        assert len(result.segments) == 2
        assert result.num_speakers == 2
        assert result.processing_time_seconds == 1.5


class TestEnhancedTranscriptionResult:
    """Test cases for EnhancedTranscriptionResult data model."""

    def test_enhanced_transcription_result_creation(self):
        """Test creating an enhanced transcription result."""
        # Create mock transcription
        transcription = TranscriptionResult(
            text="Hello world",
            confidence=0.9,
            segments=[{"start": 0.0, "end": 2.0, "text": "Hello world"}],
        )

        # Create mock diarization
        diarization = DiarizationResult(
            segments=[SpeakerSegment(0.0, 2.0, "SPEAKER_00")], num_speakers=1
        )

        # Create enhanced result
        enhanced = EnhancedTranscriptionResult(
            transcription=transcription,
            diarization=diarization,
            speaker_attributed_segments=[
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "speaker": "SPEAKER_00",
                    "confidence": 1.0,
                }
            ],
        )

        assert enhanced.transcription == transcription
        assert enhanced.diarization == diarization
        assert len(enhanced.speaker_attributed_segments) == 1
        assert enhanced.speaker_attributed_segments[0]["speaker"] == "SPEAKER_00"


class TestPyannoteAudioProcessor:
    """Test cases for PyannoteAudioProcessor."""

    def test_init(self):
        """Test processor initialization."""
        processor = PyannoteAudioProcessor(auth_token="test_token", device="cpu")

        assert processor.auth_token == "test_token"
        assert processor.device == "cpu"
        assert processor._pipeline is None

    def test_load_pipeline_success(self):
        """Test successful pipeline loading."""
        mock_pipeline = MagicMock()

        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipeline
            ),
        ):

            processor = PyannoteAudioProcessor("test_token", device="cpu")
            processor._load_pipeline()

            # Check that from_pretrained was called correctly
            from pyannote.audio import Pipeline

            Pipeline.from_pretrained.assert_called_once_with(
                "pyannote/speaker-diarization-3.1", use_auth_token="test_token"
            )
            assert processor._pipeline == mock_pipeline

    def test_load_pipeline_import_error(self):
        """Test pipeline loading with import error."""
        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "pyannote.audio.Pipeline.from_pretrained",
                side_effect=ImportError("pyannote.audio is not installed"),
            ),
        ):

            processor = PyannoteAudioProcessor("test_token", device="cpu")

            with pytest.raises(ImportError, match="pyannote.audio is not installed"):
                processor._load_pipeline()

    def test_diarize_file_not_found(self):
        """Test diarization with missing audio file."""
        processor = PyannoteAudioProcessor("test_token")

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            processor.diarize(Path("nonexistent.wav"))

    def test_diarize_success(self):
        """Test successful diarization."""
        mock_pipeline = MagicMock()

        # Mock diarization result
        mock_track = Mock()
        mock_track.start = 0.0
        mock_track.end = 2.0

        mock_pipeline.return_value.itertracks.return_value = [
            (mock_track, None, "SPEAKER_00"),
            (Mock(start=2.5, end=4.0), None, "SPEAKER_01"),
        ]

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            with (
                patch("torch.backends.mps.is_available", return_value=False),
                patch("torch.cuda.is_available", return_value=False),
                patch(
                    "pyannote.audio.Pipeline.from_pretrained",
                    return_value=mock_pipeline,
                ),
            ):

                processor = PyannoteAudioProcessor("test_token", device="cpu")
                result = processor.diarize(temp_path)

                assert isinstance(result, DiarizationResult)
                assert len(result.segments) == 2
                assert result.num_speakers == 2
                assert result.segments[0].speaker == "SPEAKER_00"
                assert result.segments[1].speaker == "SPEAKER_01"
                assert result.processing_time_seconds is not None

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_diarize_with_num_speakers_hint(self):
        """Test diarization with speaker count hint."""
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.itertracks.return_value = []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            with (
                patch("torch.backends.mps.is_available", return_value=False),
                patch("torch.cuda.is_available", return_value=False),
                patch(
                    "pyannote.audio.Pipeline.from_pretrained",
                    return_value=mock_pipeline,
                ),
            ):

                processor = PyannoteAudioProcessor("test_token", device="cpu")
                processor.diarize(temp_path, num_speakers=2)

                # Verify that min/max speakers were passed
                mock_pipeline.assert_called_with(
                    str(temp_path), min_speakers=2, max_speakers=2
                )

        finally:
            if temp_path.exists():
                os.unlink(temp_path)


class TestSegmentBasedIntegrator:
    """Test cases for SegmentBasedIntegrator."""

    def test_init(self):
        """Test integrator initialization."""
        integrator = SegmentBasedIntegrator(overlap_threshold=0.6)
        assert integrator.overlap_threshold == 0.6

    def test_init_default_threshold(self):
        """Test default overlap threshold."""
        integrator = SegmentBasedIntegrator()
        assert integrator.overlap_threshold == 0.5

    def test_integrate_simple_case(self):
        """Test integration with simple overlapping segments."""
        # Create transcription result
        transcription = TranscriptionResult(
            text="Hello world",
            confidence=0.9,
            segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello"},
                {"start": 2.5, "end": 4.0, "text": "world"},
            ],
        )

        # Create diarization result
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment(0.0, 2.5, "SPEAKER_00"),
                SpeakerSegment(2.0, 4.5, "SPEAKER_01"),
            ],
            num_speakers=2,
        )

        integrator = SegmentBasedIntegrator()
        result = integrator.integrate(transcription, diarization)

        assert isinstance(result, EnhancedTranscriptionResult)
        assert result.transcription == transcription
        assert result.diarization == diarization
        assert len(result.speaker_attributed_segments) == 2

        # Check first segment (should match SPEAKER_00)
        seg1 = result.speaker_attributed_segments[0]
        assert seg1["speaker"] == "SPEAKER_00"
        assert seg1["confidence"] == 1.0  # Perfect overlap

        # Check second segment (should match SPEAKER_01)
        seg2 = result.speaker_attributed_segments[1]
        assert seg2["speaker"] == "SPEAKER_01"
        assert seg2["confidence"] == 1.0  # Perfect overlap

    def test_integrate_no_overlap(self):
        """Test integration with no overlapping segments."""
        transcription = TranscriptionResult(
            text="Hello",
            confidence=0.9,
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )

        diarization = DiarizationResult(
            segments=[SpeakerSegment(2.0, 3.0, "SPEAKER_00")], num_speakers=1
        )

        integrator = SegmentBasedIntegrator()
        result = integrator.integrate(transcription, diarization)

        # Should have no speaker assigned due to no overlap
        seg = result.speaker_attributed_segments[0]
        assert seg["speaker"] is None
        assert seg["confidence"] == 0.0

    def test_integrate_partial_overlap_below_threshold(self):
        """Test integration with partial overlap below threshold."""
        transcription = TranscriptionResult(
            text="Hello",
            confidence=0.9,
            segments=[{"start": 0.0, "end": 2.0, "text": "Hello"}],
        )

        # Speaker segment overlaps only 0.5s out of 2.0s (25% < 50% threshold)
        diarization = DiarizationResult(
            segments=[SpeakerSegment(1.5, 3.0, "SPEAKER_00")], num_speakers=1
        )

        integrator = SegmentBasedIntegrator()
        result = integrator.integrate(transcription, diarization)

        seg = result.speaker_attributed_segments[0]
        assert seg["speaker"] is None
        assert seg["confidence"] == 0.25  # 25% overlap

    def test_integrate_invalid_segment_timing(self):
        """Test integration with invalid segment timing."""
        transcription = TranscriptionResult(
            text="Hello",
            confidence=0.9,
            segments=[
                {"start": 2.0, "end": 1.0, "text": "Hello"}
            ],  # Invalid: start > end
        )

        diarization = DiarizationResult(segments=[], num_speakers=0)

        integrator = SegmentBasedIntegrator()
        result = integrator.integrate(transcription, diarization)

        seg = result.speaker_attributed_segments[0]
        assert seg["speaker"] is None
        assert seg["confidence"] == 0.0

    def test_find_best_speaker_multiple_candidates(self):
        """Test finding best speaker among multiple candidates."""
        integrator = SegmentBasedIntegrator()

        speaker_segments = [
            SpeakerSegment(0.0, 1.0, "SPEAKER_00"),  # 0.5s overlap
            SpeakerSegment(0.5, 2.5, "SPEAKER_01"),  # 1.5s overlap (best)
            SpeakerSegment(1.8, 3.0, "SPEAKER_02"),  # 0.2s overlap
        ]

        speaker, confidence = integrator._find_best_speaker(0.5, 2.0, speaker_segments)

        assert speaker == "SPEAKER_01"
        assert confidence == 1.0  # 1.5s overlap out of 1.5s segment = 100%

    def test_find_best_speaker_empty_segments(self):
        """Test finding speaker with empty segments list."""
        integrator = SegmentBasedIntegrator()

        speaker, confidence = integrator._find_best_speaker(0.0, 1.0, [])

        assert speaker is None
        assert confidence == 0.0
