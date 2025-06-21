"""Test suite for specialized ASR (Automatic Speech Recognition) processor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from video_asr_summary.core import TranscriptionResult
from video_asr_summary.asr.funasr_specialized_processor import FunASRSpecializedProcessor


class TestFunASRSpecializedProcessor:
    """Test FunASR specialized ASR processor implementation."""
    
    def test_initialization_default_params(self):
        """Test processor initialization with default parameters."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            
            assert processor.model_path == "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            assert processor.model_revision == "v2.0.4"
            assert processor.device == "mps"
            assert processor.language == "auto"
            assert processor._model is None  # Lazy initialization
    
    def test_initialization_custom_params(self):
        """Test processor initialization with custom parameters."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRSpecializedProcessor(
                model_path="custom/asr/model",
                language="zh",
                device="cpu",
                model_revision="v1.0.0"
            )
            
            assert processor.model_path == "custom/asr/model"
            assert processor.language == "zh"
            assert processor.model_revision == "v1.0.0"
            assert processor.device == "cpu"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_device_selection_auto_mps(self, mock_cuda, mock_mps):
        """Test automatic device selection prefers MPS."""
        mock_mps.return_value = True
        mock_cuda.return_value = True
        
        with patch('torch.randn'):
            processor = FunASRSpecializedProcessor.__new__(FunASRSpecializedProcessor)
            device = processor._get_optimal_device("auto")
            
            assert device == "mps"
    
    def test_transcribe_file_not_found(self):
        """Test error handling for non-existent audio file."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            non_existent_path = Path("/non/existent/audio.wav")
            
            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                processor.transcribe(non_existent_path)
    
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
        
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            processor._initialize_model()
            
            # Verify model was initialized correctly
            mock_automodel_class.assert_called_once_with(
                model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                revision="v2.0.4",
                device="mps",
                disable_update=True,
                trust_remote_code=False,
                cache_dir=None,
            )
            assert processor._model == mock_model_instance
    
    @patch('pathlib.Path.exists')
    @patch('time.time')
    @patch('video_asr_summary.asr.funasr_specialized_processor.logger')
    def test_transcribe_success_with_spacing_fix(self, mock_logger, mock_time, mock_exists):
        """Test successful transcription with Chinese character spacing fix."""
        # Setup mocks
        mock_exists.return_value = True
        mock_time.side_effect = [0.0, 1.5]  # start, end
        
        # Mock FunASR model with character-level segments
        mock_model_instance = Mock()
        mock_asr_result = [{
            "text": "你 好 世 界 ， 这 是 一 个 测 试 。",  # Spaced Chinese
            "timestamp": [
                [0, 200], [200, 400], [400, 600], [600, 800], [800, 1000],  # 你 好 世 界 ，
                [1000, 1200], [1200, 1400], [1400, 1600], [1600, 1800], [1800, 2000],  # 这 是 一 个 测
                [2000, 2200], [2200, 2400], [2400, 2600]  # 试 。
            ]
        }]
        mock_model_instance.generate.return_value = mock_asr_result
        
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            # Mock the model directly
            processor._model = mock_model_instance
            
            audio_path = Path("/test/audio.wav")
            result = processor.transcribe(audio_path)
            
            # Verify result structure
            assert isinstance(result, TranscriptionResult)
            assert result.text == "你好世界，这是一个测试。"  # Spaces removed
            assert result.processing_time_seconds == 1.5
            assert len(result.segments) > 0  # Should have combined segments
            assert result.language == "zh"  # Detected as Chinese
            assert 0.0 <= result.confidence <= 1.0
    
    @patch('pathlib.Path.exists')  
    @patch('time.time')
    def test_transcribe_english_no_spacing_change(self, mock_time, mock_exists):
        """Test transcription preserves English spacing."""
        # Setup mocks
        mock_exists.return_value = True
        mock_time.side_effect = [0.0, 1.0]
        
        # Mock FunASR model with word-level segments for English
        mock_model_instance = Mock()
        mock_asr_result = [{
            "text": "Hello world, this is a test.",
            "timestamp": [
                [0, 200], [200, 400], [400, 600], [600, 800], [800, 1000], [1000, 1200]  # Each word/token
            ]
        }]
        mock_model_instance.generate.return_value = mock_asr_result

        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            # Mock the model directly
            processor._model = mock_model_instance
            
            audio_path = Path("/test/audio.wav")
            result = processor.transcribe(audio_path)
            
            # Verify English text maintains proper word spacing
            assert "Hello" in result.text and "world" in result.text
            assert result.language == "en"
    
    def test_character_level_processing(self):
        """Test that the new character-level processing works correctly."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            
            # Test character splitting
            char_pieces = processor._split_text_to_characters("你 好 世 界")
            assert char_pieces == ["你", "好", "世", "界"]
            
            # Test English word splitting
            english_pieces = processor._split_text_to_characters("Hello world test")
            assert english_pieces == ["Hello", "world", "test"]
    
    def test_detect_language_chinese(self):
        """Test language detection for Chinese text."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            
            # Test Chinese detection
            chinese_text = "你好世界，这是中文测试。"
            result = processor._detect_language(chinese_text, "auto")
            assert result == "zh"
    
    def test_detect_language_english(self):
        """Test language detection for English text."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            
            # Test English detection
            english_text = "Hello world, this is an English test."
            result = processor._detect_language(english_text, "auto")
            assert result == "en"
    
    def test_detect_language_explicit_setting(self):
        """Test language detection with explicit setting."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            
            # Test explicit language setting overrides detection
            text = "Any text"
            result = processor._detect_language(text, "zh")
            assert result == "zh"
    
    def test_lazy_model_initialization(self):
        """Test that model is only initialized when needed."""
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
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
        
        with patch.object(FunASRSpecializedProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRSpecializedProcessor()
            
            # Call multiple times
            processor._initialize_model()
            processor._initialize_model()
            processor._initialize_model()
            
            # Should only be called once
            mock_automodel_class.assert_called_once()
