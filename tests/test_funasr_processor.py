"""Tests for FunASR processor with GPU acceleration."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

from video_asr_summary.asr.funasr_processor import FunASRProcessor
from video_asr_summary.core import TranscriptionResult


class TestFunASRProcessor:
    """Test suite for FunASR processor with actual model testing."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use environment variable for test audio, fallback to None
        test_audio_env = os.getenv('FUNASR_TEST_AUDIO_PATH')
        if test_audio_env:
            self.test_audio_path = Path(test_audio_env)
        else:
            # Create a synthetic test audio file for integration tests
            self.test_audio_path = self._create_test_audio_file()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up synthetic audio file if we created it
        if hasattr(self, '_temp_audio_file') and self._temp_audio_file.exists():
            self._temp_audio_file.unlink()
    
    def _create_test_audio_file(self) -> Union[Path, None]:
        """Create a synthetic test audio file for integration tests."""
        try:
            import numpy as np
            from scipy.io import wavfile
            
            # Create a temporary audio file with synthetic content
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = Path(temp_file.name)
            temp_file.close()
            
            # Generate 3 seconds of synthetic audio (sine wave)
            sample_rate = 16000
            duration = 3.0
            frequency = 440  # A4 note
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            # Create a simple tone that will transcribe to something recognizable
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Convert to 16-bit PCM
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Save as WAV file
            wavfile.write(temp_path, sample_rate, audio_data)
            
            self._temp_audio_file = temp_path
            return temp_path
            
        except ImportError:
            # If scipy/numpy not available, return None to skip integration tests
            return None
        
    def test_device_explicit_setting(self):
        """Test explicit device setting."""
        with patch.object(FunASRProcessor, '_get_optimal_device', side_effect=lambda x: x):
            processor_cpu = FunASRProcessor(device="cpu")
            assert processor_cpu.device == "cpu"
            
            processor_mps = FunASRProcessor(device="mps")
            assert processor_mps.device == "mps"
            
            processor_cuda = FunASRProcessor(device="cuda")
            assert processor_cuda.device == "cuda"
        
    def test_device_auto_selection(self):
        """Test automatic device selection logic."""
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="mps") as mock_get_device:
            processor = FunASRProcessor(device="auto")
            mock_get_device.assert_called_once_with("auto")
            assert processor.device == "mps"
        
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu") as mock_get_device:
            processor = FunASRProcessor(device="auto")
            assert processor.device == "cpu"
        
    def test_model_initialization_parameters(self):
        """Test FunASR processor initialization with different parameters."""
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            # Test default parameters
            processor_default = FunASRProcessor()
            assert processor_default.model_path == "iic/SenseVoiceSmall"
            assert processor_default.language == "auto"
            assert processor_default.model_revision == "main"
            assert not processor_default.suppress_warnings
            
            # Test custom parameters
            processor_custom = FunASRProcessor(
                model_path="custom/model",
                language="zn", 
                device="cpu",
                model_revision="v1.0",
                suppress_warnings=True
            )
            assert processor_custom.model_path == "custom/model"
            assert processor_custom.language == "zn"
            assert processor_custom.device == "cpu"
            assert processor_custom.model_revision == "v1.0"
            assert processor_custom.suppress_warnings
        
    def test_model_initialization_with_revision(self):
        """Test FunASR processor initialization with model revision."""
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(
                model_path="iic/SenseVoiceSmall",
                model_revision="main",
                device="cpu"
            )
            assert processor.model_path == "iic/SenseVoiceSmall"
            assert processor.model_revision == "main"
            assert processor.device == "cpu"
        
    def test_lazy_model_initialization(self):
        """Test that model is only initialized when needed."""
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(device="cpu")
            assert processor._model is None
        
    def test_transcribe_file_not_found(self):
        """Test transcribe raises error for non-existent file."""
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(device="cpu")
            non_existent_path = Path("/non/existent/audio.wav")
            
            with pytest.raises(FileNotFoundError):
                processor.transcribe(non_existent_path)

    @pytest.mark.integration
    def test_transcribe_with_cpu_device(self):
        """Integration test: transcribe with CPU device."""
        if self.test_audio_path is None:
            pytest.skip("No test audio file available. Set FUNASR_TEST_AUDIO_PATH environment variable or install scipy/numpy for synthetic audio generation.")
        
        if not self.test_audio_path.exists():
            pytest.skip(f"Test audio file not found: {self.test_audio_path}")
            
        processor = FunASRProcessor(device="cpu")
        result = processor.transcribe(self.test_audio_path)
        
        self._validate_transcription_result(result)
        # Note: Synthetic audio may not contain recognizable text, so we just validate structure
        
    @pytest.mark.integration  
    @pytest.mark.skipif(not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available(), 
                       reason="MPS not available")
    def test_transcribe_with_mps_device(self):
        """Integration test: transcribe with MPS device."""
        if self.test_audio_path is None:
            pytest.skip("No test audio file available. Set FUNASR_TEST_AUDIO_PATH environment variable or install scipy/numpy for synthetic audio generation.")
        
        if not self.test_audio_path.exists():
            pytest.skip(f"Test audio file not found: {self.test_audio_path}")
            
        processor = FunASRProcessor(device="mps")
        result = processor.transcribe(self.test_audio_path)
        
        self._validate_transcription_result(result)
        # Note: Synthetic audio may not contain recognizable text, so we just validate structure
        
    @pytest.mark.integration
    def test_transcribe_with_auto_device(self):
        """Integration test: transcribe with auto device detection."""
        if self.test_audio_path is None:
            pytest.skip("No test audio file available. Set FUNASR_TEST_AUDIO_PATH environment variable or install scipy/numpy for synthetic audio generation.")
        
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
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_mps_available(self, mock_cuda_available, mock_mps_available):
        """Test device selection when MPS is available."""
        mock_mps_available.return_value = True
        mock_cuda_available.return_value = False
        
        with patch('torch.device'), patch('torch.randn'):
            processor = FunASRProcessor.__new__(FunASRProcessor)  # Create without __init__
            result = processor._get_optimal_device("auto")
            assert result == "mps"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_cuda_available(self, mock_cuda_available, mock_mps_available):
        """Test device selection when CUDA is available but not MPS."""
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = True
        
        processor = FunASRProcessor.__new__(FunASRProcessor)
        result = processor._get_optimal_device("auto")
        assert result == "cuda"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_get_optimal_device_cpu_fallback(self, mock_cuda_available, mock_mps_available):
        """Test device selection falls back to CPU."""
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = False
        
        processor = FunASRProcessor.__new__(FunASRProcessor)
        result = processor._get_optimal_device("auto")
        assert result == "cpu"
    
    def test_get_optimal_device_explicit_device(self):
        """Test device selection with explicit device specification."""
        processor = FunASRProcessor.__new__(FunASRProcessor)
        
        assert processor._get_optimal_device("cpu") == "cpu"
        assert processor._get_optimal_device("mps") == "mps"
        assert processor._get_optimal_device("cuda") == "cuda"
        assert processor._get_optimal_device("cuda:0") == "cuda:0"
    
    @patch('builtins.__import__', side_effect=ImportError("No torch"))
    def test_get_optimal_device_import_error(self, mock_import):
        """Test device selection handles torch import errors."""
        processor = FunASRProcessor.__new__(FunASRProcessor)
        result = processor._get_optimal_device("auto")
        assert result == "cpu"
    
    @patch('builtins.__import__')
    def test_initialize_model_success(self, mock_import):
        """Test successful model initialization."""
        # Mock the FunASR AutoModel import
        mock_funasr_module = Mock()
        mock_automodel = Mock()
        mock_model = Mock()
        mock_automodel.return_value = mock_model
        mock_funasr_module.AutoModel = mock_automodel
        
        def side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(device="cpu")
            processor._initialize_model()
            
            mock_automodel.assert_called_once_with(
                model="iic/SenseVoiceSmall",
                revision="main",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cpu",
                disable_update=True,
                trust_remote_code=False,
                cache_dir=None,
            )
            assert processor._model == mock_model
    
    @patch('builtins.__import__', side_effect=ImportError("No FunASR"))
    def test_initialize_model_import_error(self, mock_import):
        """Test model initialization raises ImportError when FunASR not available."""
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(device="cpu")
            
            with pytest.raises(ImportError, match="FunASR is not installed"):
                processor._initialize_model()
    
    @patch('builtins.__import__')
    def test_initialize_model_only_once(self, mock_import):
        """Test model is only initialized once."""
        # Mock the FunASR AutoModel import
        mock_funasr_module = Mock()
        mock_automodel = Mock()
        mock_model = Mock()
        mock_automodel.return_value = mock_model
        mock_funasr_module.AutoModel = mock_automodel
        
        def side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(device="cpu")
            
            # Call multiple times
            processor._initialize_model()
            processor._initialize_model()
            processor._initialize_model()
            
            # Should only be called once
            mock_automodel.assert_called_once()
    
    @patch('builtins.__import__')
    def test_initialize_model_failure(self, mock_import):
        """Test model initialization handles failures."""
        # Mock the FunASR AutoModel import to fail
        mock_funasr_module = Mock()
        mock_automodel = Mock(side_effect=Exception("Model init failed"))
        mock_funasr_module.AutoModel = mock_automodel
        
        def side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(device="cpu")
            
            with pytest.raises(Exception, match="Failed to initialize FunASR model"):
                processor._initialize_model()
    
    @patch('pathlib.Path.exists')
    @patch('builtins.__import__')
    @patch('time.time')
    def test_transcribe_success_mocked(self, mock_time, mock_import, mock_exists):
        """Test successful transcription with mocked dependencies."""
        # Setup path mocks
        mock_exists.return_value = True
        mock_time.side_effect = [0.0, 1.5, 3.0]  # start, after_generate, end
        
        # Mock the FunASR imports
        mock_funasr_module = Mock()
        mock_automodel = Mock()
        mock_model = Mock()
        mock_automodel.return_value = mock_model
        mock_funasr_module.AutoModel = mock_automodel
        
        # Mock rich_transcription_postprocess
        mock_postprocess_module = Mock()
        mock_postprocess = Mock(return_value="测试音频文本内容")
        mock_postprocess_module.rich_transcription_postprocess = mock_postprocess
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            elif name == 'funasr.utils.postprocess_utils':
                return mock_postprocess_module
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = import_side_effect
        
        # Mock FunASR result
        mock_funasr_result = [{
            "text": "测试音频文本内容",
            "timestamp": [
                [0, 1500, "测试音频"],
                [1500, 3000, "文本内容"]
            ]
        }]
        mock_model.generate.return_value = mock_funasr_result
        
        with patch.object(FunASRProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRProcessor(device="cpu")
            
            audio_path = Path("/test/audio.wav")
            result = processor.transcribe(audio_path)
            
            # Verify result
            assert isinstance(result, TranscriptionResult)
            assert result.text == "测试音频文本内容"
            assert result.processing_time_seconds == 1.5
            assert len(result.segments) == 2
            assert result.language == "zh"
            assert 0.0 <= result.confidence <= 1.0
            
            # Verify mocks were called correctly
            mock_model.generate.assert_called_once_with(
                input=str(audio_path),
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            mock_postprocess.assert_called_once()
