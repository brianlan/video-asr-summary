"""Tests for FunASR processor with GPU acceleration."""

import pytest
import torch
from pathlib import Path

from video_asr_summary.asr.funasr_processor import FunASRProcessor
from video_asr_summary.core import TranscriptionResult


class TestFunASRProcessor:
    """Test suite for FunASR processor with actual model testing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_audio_path = Path("/Users/rlan/Downloads/ruige-huangjin-4000/audio.wav")
        
    def test_device_explicit_setting(self):
        """Test explicit device setting."""
        processor_cpu = FunASRProcessor(device="cpu")
        assert processor_cpu.device == "cpu"
        
        processor_mps = FunASRProcessor(device="mps")
        assert processor_mps.device == "mps"
        
        processor_auto = FunASRProcessor(device="auto")
        # Auto should select mps on Apple Silicon, cpu otherwise
        assert processor_auto.device in ["mps", "cpu"]
        
    def test_model_initialization_parameters(self):
        """Test FunASR processor initialization with different parameters."""
        # Test default parameters
        processor_default = FunASRProcessor()
        assert processor_default.model_path == "iic/SenseVoiceSmall"
        assert processor_default.language == "auto"
        
        # Test custom parameters
        processor_custom = FunASRProcessor(
            model_path="custom/model",
            language="zn", 
            device="cpu"
        )
        assert processor_custom.model_path == "custom/model"
        assert processor_custom.language == "zn"
        assert processor_custom.device == "cpu"
        
    def test_lazy_model_initialization(self):
        """Test that model is only initialized when needed."""
        processor = FunASRProcessor(device="cpu")
        assert processor._model is None
        
    def test_transcribe_file_not_found(self):
        """Test transcribe raises error for non-existent file."""
        processor = FunASRProcessor(device="cpu")
        non_existent_path = Path("/non/existent/audio.wav")
        
        with pytest.raises(FileNotFoundError):
            processor.transcribe(non_existent_path)

    @pytest.mark.integration
    def test_transcribe_with_cpu_device(self):
        """Integration test: transcribe with CPU device."""
        if not self.test_audio_path.exists():
            pytest.skip(f"Test audio file not found: {self.test_audio_path}")
            
        processor = FunASRProcessor(device="cpu")
        result = processor.transcribe(self.test_audio_path)
        
        self._validate_transcription_result(result)
        assert "黄金" in result.text  # Should contain expected Chinese content
        
    @pytest.mark.integration  
    @pytest.mark.skipif(not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available(), 
                       reason="MPS not available")
    def test_transcribe_with_mps_device(self):
        """Integration test: transcribe with MPS device."""
        if not self.test_audio_path.exists():
            pytest.skip(f"Test audio file not found: {self.test_audio_path}")
            
        processor = FunASRProcessor(device="mps")
        result = processor.transcribe(self.test_audio_path)
        
        self._validate_transcription_result(result)
        assert "黄金" in result.text  # Should contain expected Chinese content
        
    @pytest.mark.integration
    def test_transcribe_with_auto_device(self):
        """Integration test: transcribe with auto device detection."""
        if not self.test_audio_path.exists():
            pytest.skip(f"Test audio file not found: {self.test_audio_path}")
            
        processor = FunASRProcessor(device="auto")
        result = processor.transcribe(self.test_audio_path)
        
        self._validate_transcription_result(result)
        # Verify device was auto-selected properly
        expected_device = "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu"
        assert processor.device == expected_device
        
    def _validate_transcription_result(self, result: TranscriptionResult):
        """Validate that transcription result meets expected criteria."""
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.segments, list)
        assert len(result.segments) > 0
        assert isinstance(result.processing_time_seconds, float)
        assert result.processing_time_seconds > 0
        
        # Validate segment structure
        for segment in result.segments:
            assert isinstance(segment, dict)
            assert "id" in segment
            assert "start" in segment  
            assert "end" in segment
            assert "text" in segment
            assert "confidence" in segment
            assert isinstance(segment["start"], (int, float))
            assert isinstance(segment["end"], (int, float))
            assert segment["start"] >= 0
            assert segment["end"] >= segment["start"]
