"""Test suite for Punctuation Restoration processor."""

from unittest.mock import Mock, patch
import builtins

from video_asr_summary.core import PunctuationResult
from video_asr_summary.punctuation.funasr_punc_processor import FunASRPunctuationProcessor


class TestFunASRPunctuationProcessor:
    """Test FunASR punctuation restoration processor implementation."""
    
    def test_initialization_default_params(self):
        """Test processor initialization with default parameters."""
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            
            assert processor.model_path == "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
            assert processor.model_revision == "v2.0.4"
            assert processor.device == "mps"
            assert processor._model is None  # Lazy initialization
    
    def test_initialization_custom_params(self):
        """Test processor initialization with custom parameters."""
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="cpu"):
            processor = FunASRPunctuationProcessor(
                model_path="custom/punc/model",
                device="cpu",
                model_revision="v1.0.0"
            )
            
            assert processor.model_path == "custom/punc/model"
            assert processor.model_revision == "v1.0.0"
            assert processor.device == "cpu"
    
    @patch('torch.backends.mps.is_available')
    @patch('torch.cuda.is_available')
    def test_device_selection_auto_mps(self, mock_cuda, mock_mps):
        """Test automatic device selection prefers MPS."""
        mock_mps.return_value = True
        mock_cuda.return_value = True
        
        with patch('torch.randn'):
            processor = FunASRPunctuationProcessor.__new__(FunASRPunctuationProcessor)
            device = processor._get_optimal_device("auto")
            
            assert device == "mps"
    
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
            return builtins.__import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            processor._initialize_model()
            
            # Verify model was initialized correctly
            mock_automodel_class.assert_called_once_with(
                model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                model_revision="v2.0.4",
                device="mps",
                disable_update=True,
                trust_remote_code=False,
                cache_dir=None,
            )
            assert processor._model == mock_model_instance
    
    @patch('builtins.__import__')
    @patch('time.time')
    def test_restore_punctuation_success(self, mock_time, mock_import):
        """Test successful punctuation restoration."""
        # Setup mocks
        mock_time.side_effect = [0.0, 1.0]  # start, end
        
        # Mock FunASR
        mock_funasr_module = Mock()
        mock_automodel_class = Mock()
        mock_model_instance = Mock()
        
        # Mock punctuation result
        mock_punc_result = [{
            "text": "你好世界，这是一个测试。"  # Text with restored punctuation
        }]
        mock_model_instance.generate.return_value = mock_punc_result
        mock_automodel_class.return_value = mock_model_instance
        mock_funasr_module.AutoModel = mock_automodel_class
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return builtins.__import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            input_text = "你好世界这是一个测试"  # Text without punctuation
            result = processor.restore_punctuation(input_text)
            
            # Verify result structure
            assert isinstance(result, PunctuationResult)
            assert result.text == "你好世界，这是一个测试。"  # Punctuation restored
            assert result.processing_time_seconds == 1.0
            assert 0.0 <= result.confidence <= 1.0
            
            # Verify model was called correctly
            mock_model_instance.generate.assert_called_once_with(
                input=input_text,
                cache={},
                language="auto",
                use_itn=False,
                batch_size_s=60,
            )
    
    @patch('builtins.__import__')
    def test_restore_punctuation_empty_text(self, mock_import):
        """Test punctuation restoration with empty text."""
        # Mock FunASR
        mock_funasr_module = Mock()
        mock_automodel_class = Mock()
        mock_model_instance = Mock()
        mock_automodel_class.return_value = mock_model_instance
        mock_funasr_module.AutoModel = mock_automodel_class
        
        def import_side_effect(name, *args, **kwargs):
            if name == 'funasr':
                return mock_funasr_module
            return builtins.__import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            result = processor.restore_punctuation("")
            
            # Should return empty result without calling model
            assert isinstance(result, PunctuationResult)
            assert result.text == ""
            assert result.confidence == 0.0
            assert result.processing_time_seconds is not None
            
            # Model should not be called for empty text
            mock_model_instance.generate.assert_not_called()
    
    def test_estimate_confidence_with_punctuation(self):
        """Test confidence estimation with good punctuation."""
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            
            # Text with good punctuation should have high confidence
            text_with_punct = "你好世界，这是一个测试。"
            confidence = processor._estimate_confidence(text_with_punct)
            assert confidence >= 0.9
    
    def test_estimate_confidence_without_punctuation(self):
        """Test confidence estimation without punctuation."""
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            
            # Text without punctuation should have lower confidence
            text_without_punct = "你好世界这是一个测试"
            confidence = processor._estimate_confidence(text_without_punct)
            assert confidence < 0.9
    
    def test_estimate_confidence_empty_text(self):
        """Test confidence estimation with empty text."""
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            
            confidence = processor._estimate_confidence("")
            assert confidence == 0.0
    
    def test_lazy_model_initialization(self):
        """Test that model is only initialized when needed."""
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
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
            return builtins.__import__(name, *args, **kwargs)
        
        mock_import.side_effect = import_side_effect
        
        with patch.object(FunASRPunctuationProcessor, '_get_optimal_device', return_value="mps"):
            processor = FunASRPunctuationProcessor()
            
            # Call multiple times
            processor._initialize_model()
            processor._initialize_model()
            processor._initialize_model()
            
            # Should only be called once
            mock_automodel_class.assert_called_once()
