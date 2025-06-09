"""Tests for ASR processors."""

from pathlib import Path
from unittest.mock import patch

import pytest

from video_asr_summary.asr.whisper_processor import WhisperProcessor
from video_asr_summary.core import TranscriptionResult


class TestWhisperProcessor:
    """Tests for WhisperProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = WhisperProcessor()
        self.test_audio_path = Path("/test/audio.wav")

    def test_init_default_model(self):
        """Test WhisperProcessor initialization with default model."""
        processor = WhisperProcessor()
        assert processor.model_name == "mlx-community/whisper-large-v3-turbo"
        assert processor.language is None

    def test_init_custom_model_and_language(self):
        """Test WhisperProcessor initialization with custom model and language."""
        processor = WhisperProcessor(
            model_name="mlx-community/whisper-base", language="en"
        )
        assert processor.model_name == "mlx-community/whisper-base"
        assert processor.language == "en"

    @patch("video_asr_summary.asr.whisper_processor.mlx_whisper")
    @patch("pathlib.Path.exists")
    def test_transcribe_success(self, mock_exists, mock_mlx_whisper):
        """Test successful transcription."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock mlx_whisper.transcribe response
        mock_result = {
            "text": "This is a test transcription.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "This is a test",
                    "avg_logprob": -0.5,
                    "no_speech_prob": 0.1,
                },
                {
                    "start": 2.5,
                    "end": 5.0,
                    "text": " transcription.",
                    "avg_logprob": -0.3,
                    "no_speech_prob": 0.05,
                },
            ],
            "language": "en",
        }
        mock_mlx_whisper.transcribe.return_value = mock_result

        # Execute transcription
        result = self.processor.transcribe(self.test_audio_path)

        # Verify the call
        mock_mlx_whisper.transcribe.assert_called_once_with(
            str(self.test_audio_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language=None,
        )

        # Verify the result
        assert isinstance(result, TranscriptionResult)
        assert result.text == "This is a test transcription."
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.confidence > 0.0
        assert result.processing_time_seconds is not None

    @patch("video_asr_summary.asr.whisper_processor.mlx_whisper")
    @patch("pathlib.Path.exists")
    def test_transcribe_with_language_hint(self, mock_exists, mock_mlx_whisper):
        """Test transcription with language hint."""
        # Mock file existence
        mock_exists.return_value = True

        processor = WhisperProcessor(language="zh")
        mock_result = {
            "text": "这是一个测试转录。",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "这是一个测试转录。",
                    "avg_logprob": -0.4,
                    "no_speech_prob": 0.1,
                }
            ],
            "language": "zh",
        }
        mock_mlx_whisper.transcribe.return_value = mock_result

        result = processor.transcribe(self.test_audio_path)

        mock_mlx_whisper.transcribe.assert_called_once_with(
            str(self.test_audio_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="zh",
        )
        assert result.text == "这是一个测试转录。"
        assert result.language == "zh"

    def test_transcribe_file_not_found(self):
        """Test transcription with non-existent file."""
        non_existent_path = Path("/non/existent/file.wav")

        with pytest.raises(FileNotFoundError):
            self.processor.transcribe(non_existent_path)

    @patch("video_asr_summary.asr.whisper_processor.mlx_whisper")
    @patch("pathlib.Path.exists")
    def test_transcribe_mlx_whisper_error(self, mock_exists, mock_mlx_whisper):
        """Test handling of mlx_whisper errors."""
        # Mock file existence
        mock_exists.return_value = True

        mock_mlx_whisper.transcribe.side_effect = Exception("MLX Whisper error")

        with pytest.raises(Exception) as exc_info:
            self.processor.transcribe(self.test_audio_path)

        assert "MLX Whisper error" in str(exc_info.value)

    @patch("video_asr_summary.asr.whisper_processor.mlx_whisper")
    @patch("pathlib.Path.exists")
    def test_calculate_confidence_score(self, mock_exists, mock_mlx_whisper):
        """Test confidence score calculation from segments."""
        # Mock file existence
        mock_exists.return_value = True

        mock_result = {
            "text": "Test transcription.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Test",
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.1,
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": " transcription.",
                    "avg_logprob": -0.4,
                    "no_speech_prob": 0.05,
                },
            ],
            "language": "en",
        }
        mock_mlx_whisper.transcribe.return_value = mock_result

        result = self.processor.transcribe(self.test_audio_path)

        # Confidence should be calculated based on avg_logprob and no_speech_prob
        # Expected confidence should be reasonable (> 0.5 for good quality)
        assert 0.0 < result.confidence <= 1.0

    @patch("video_asr_summary.asr.whisper_processor.mlx_whisper")
    @patch("pathlib.Path.exists")
    def test_empty_segments_handling(self, mock_exists, mock_mlx_whisper):
        """Test handling of empty segments."""
        # Mock file existence
        mock_exists.return_value = True

        mock_result = {"text": "", "segments": [], "language": "en"}
        mock_mlx_whisper.transcribe.return_value = mock_result

        result = self.processor.transcribe(self.test_audio_path)

        assert result.text == ""
        assert result.segments == []
        assert result.confidence == 0.0

    @patch("video_asr_summary.asr.whisper_processor.mlx_whisper")
    @patch("pathlib.Path.exists")
    def test_segments_structure_preserved(self, mock_exists, mock_mlx_whisper):
        """Test that segment structure is properly preserved."""
        # Mock file existence
        mock_exists.return_value = True

        mock_result = {
            "text": "Hello world.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Hello world.",
                    "avg_logprob": -0.3,
                    "no_speech_prob": 0.05,
                    "tokens": [123, 456, 789],
                    "temperature": 0.0,
                }
            ],
            "language": "en",
        }
        mock_mlx_whisper.transcribe.return_value = mock_result

        result = self.processor.transcribe(self.test_audio_path)

        assert len(result.segments) == 1
        segment = result.segments[0]
        assert segment["start"] == 0.0
        assert segment["end"] == 1.5
        assert segment["text"] == "Hello world."
        assert "avg_logprob" in segment
        assert "no_speech_prob" in segment
