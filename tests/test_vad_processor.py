"""Test suite for VAD (Voice Activity Detection) processor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from video_asr_summary.core import VADResult, VADSegment
from video_asr_summary.vad.funasr_vad_processor import FunASRVADProcessor


class TestVADTypes:
    """Test VAD data types."""
    
    def test_vad_segment_creation(self):
        """Test VADSegment data class creation."""
        segment = VADSegment(start=1.0, end=2.5, confidence=0.95)
        
        assert segment.start == 1.0
        assert segment.end == 2.5
        assert segment.confidence == 0.95
    
    def test_vad_segment_defaults(self):
        """Test VADSegment default values."""
        segment = VADSegment(start=0.0, end=1.0)
        
        assert segment.confidence == 1.0  # Default confidence
    
    def test_vad_result_creation(self):
        """Test VADResult data class creation."""
        segments = [
            VADSegment(start=0.0, end=1.5, confidence=0.9),
            VADSegment(start=2.0, end=3.5, confidence=0.85)
        ]
        result = VADResult(
            segments=segments,
            total_speech_duration=3.0,
            processing_time_seconds=0.5
        )
        
        assert len(result.segments) == 2
        assert result.total_speech_duration == 3.0
        assert result.processing_time_seconds == 0.5


class TestFunASRVADProcessor:
    """Test FunASR VAD processor implementation."""
    
    def test_initialization_default_params(self):
        """Test processor initialization with default parameters."""
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            
            assert processor.model_path == "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            assert processor.model_revision == "v2.0.4"
            assert processor.device == "cpu"
            assert processor._model is None  # Lazy initialization
    
    def test_initialization_custom_params(self):
        """Test processor initialization with custom parameters."""
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRVADProcessor(
                model_path="custom/vad/model",
                device="mps",
                model_revision="v1.0.0"
            )
            
            assert processor.model_path == "custom/vad/model"
            assert processor.model_revision == "v1.0.0"
            assert processor.device == "mps"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_device_selection_auto_mps(self, mock_cuda, mock_mps):
        """Test automatic device selection prefers MPS."""
        mock_mps.return_value = True
        mock_cuda.return_value = True
        
        with patch('torch.randn'):
            processor = FunASRVADProcessor.__new__(FunASRVADProcessor)
            device = processor._get_optimal_device("auto")
            
            assert device == "mps"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_device_selection_auto_cuda(self, mock_cuda, mock_mps):
        """Test automatic device selection falls back to CUDA."""
        mock_mps.return_value = False
        mock_cuda.return_value = True
        
        processor = FunASRVADProcessor.__new__(FunASRVADProcessor)
        device = processor._get_optimal_device("auto")
        
        assert device == "cuda"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_device_selection_auto_cpu(self, mock_cuda, mock_mps):
        """Test automatic device selection falls back to CPU."""
        mock_mps.return_value = False
        mock_cuda.return_value = False
        
        processor = FunASRVADProcessor.__new__(FunASRVADProcessor)
        device = processor._get_optimal_device("auto")
        
        assert device == "cpu"
    
    def test_device_selection_explicit(self):
        """Test explicit device selection."""
        processor = FunASRVADProcessor.__new__(FunASRVADProcessor)
        
        assert processor._get_optimal_device("cpu") == "cpu"
        assert processor._get_optimal_device("mps") == "mps"
        assert processor._get_optimal_device("cuda") == "cuda"
    
    def test_detect_voice_activity_file_not_found(self):
        """Test error handling for non-existent audio file."""
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            non_existent_path = Path("/non/existent/audio.wav")
            
            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                processor.detect_voice_activity(non_existent_path)
    
    @patch('builtins.__import__')
    def test_model_initialization_success(self, mock_import):
        """Test successful model initialization."""
        # Mock FunASR imports
        mock_funasr_module = Mock()
        mock_automodel_class = Mock()
        mock_model_instance = Mock()
        mock_automodel_class.return_value = mock_model_instance
        mock_funasr_module.AutoModel = mock_automodel_class
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            processor._initialize_model()
            
            # Verify model was initialized correctly
            mock_automodel_class.assert_called_once_with(
                model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                model_revision="v2.0.4",
                device="cpu",
                disable_update=True,
                trust_remote_code=False,
                cache_dir=None,
            )
            assert processor._model == mock_model_instance
    
    @patch('builtins.__import__')
    def test_model_initialization_import_error(self, mock_import):
        """Test model initialization handles import errors."""
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                raise ImportError("FunASR not found")
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            
            with pytest.raises(ImportError, match="FunASR is not installed"):
                processor._initialize_model()
    
    @patch('pathlib.Path.exists')
    @patch('builtins.__import__')
    @patch('time.time')
    def test_detect_voice_activity_success(self, mock_time, mock_import, mock_exists):
        """Test successful voice activity detection."""
        # Setup mocks
        mock_exists.return_value = True
        mock_time.side_effect = [0.0, 1.5]  # start, end
        
        # Mock FunASR
        mock_funasr_module = Mock()
        mock_automodel_class = Mock()
        mock_model_instance = Mock()
        
        # Mock VAD result
        mock_vad_result = [{
            "value": [
                [0, 1500],      # 0-1.5 seconds speech
                [2000, 3500],   # 2-3.5 seconds speech  
                [5000, 6000],   # 5-6 seconds speech
            ]
        }]
        mock_model_instance.generate.return_value = mock_vad_result
        mock_automodel_class.return_value = mock_model_instance
        mock_funasr_module.AutoModel = mock_automodel_class
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            audio_path = Path("/test/audio.wav")
            result = processor.detect_voice_activity(audio_path)
            
            # Verify result structure
            assert isinstance(result, VADResult)
            assert len(result.segments) == 3
            assert result.processing_time_seconds == 1.5
            assert result.total_speech_duration == 4.0  # 1.5 + 1.5 + 1.0
            
            # Verify segments
            segments = result.segments
            assert segments[0].start == 0.0
            assert segments[0].end == 1.5
            assert segments[1].start == 2.0
            assert segments[1].end == 3.5
            assert segments[2].start == 5.0
            assert segments[2].end == 6.0
            
            # Verify model was called correctly
            mock_model_instance.generate.assert_called_once_with(
                input=str(audio_path),
                cache={},
                language="auto",
                use_itn=False,
                batch_size_s=60,
            )
    
    @patch('pathlib.Path.exists')
    @patch('builtins.__import__')
    def test_detect_voice_activity_no_speech(self, mock_import, mock_exists):
        """Test voice activity detection with no speech detected."""
        mock_exists.return_value = True
        
        # Mock FunASR
        mock_funasr_module = Mock()
        mock_automodel_class = Mock()
        mock_model_instance = Mock()
        
        # Mock empty VAD result
        mock_vad_result = [{"value": []}]
        mock_model_instance.generate.return_value = mock_vad_result
        mock_automodel_class.return_value = mock_model_instance
        mock_funasr_module.AutoModel = mock_automodel_class
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            audio_path = Path("/test/audio.wav")
            result = processor.detect_voice_activity(audio_path)
            
            assert isinstance(result, VADResult)
            assert len(result.segments) == 0
            assert result.total_speech_duration == 0.0
    
    def test_lazy_model_initialization(self):
        """Test that model is only initialized when needed."""
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            assert processor._model is None
    
    @patch('builtins.__import__')
    def test_model_initialized_only_once(self, mock_import):
        """Test that model initialization happens only once."""
        mock_funasr_module = Mock()
        mock_automodel_class = Mock()
        mock_model_instance = Mock()
        mock_automodel_class.return_value = mock_model_instance
        mock_funasr_module.AutoModel = mock_automodel_class
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRVADProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRVADProcessor()
            
            # Call multiple times
            processor._initialize_model()
            processor._initialize_model()
            processor._initialize_model()
            
            # Should only be called once
            mock_automodel_class.assert_called_once()
