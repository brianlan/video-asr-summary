"""Test language-aware ASR processor selection."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator


class TestLanguageAwareASR:
    """Test language-aware ASR processor selection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.orchestrator = PipelineOrchestrator(str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_should_use_specialized_asr_for_chinese(self):
        """Test that SpecializedASRIntegrator is used for Chinese language."""
        # When: Check ASR processor for Chinese
        with patch('video_asr_summary.pipeline.orchestrator.SpecializedASRIntegrator') as mock_specialized:
            mock_specialized.return_value = Mock()
            
            processor = self.orchestrator._get_asr_processor('zh')
            
            # Then: Should use SpecializedASRIntegrator
            mock_specialized.assert_called_once_with(device="auto")
            assert processor is not None

    def test_should_use_whisper_for_english(self):
        """Test that WhisperProcessor is used for English language."""
        # When: Check ASR processor for English
        with patch('video_asr_summary.pipeline.orchestrator.WhisperProcessor') as mock_whisper:
            mock_whisper.return_value = Mock()
            
            processor = self.orchestrator._get_asr_processor('en')
            
            # Then: Should use WhisperProcessor
            mock_whisper.assert_called_once_with(language="en")
            assert processor is not None

    def test_should_use_whisper_for_other_languages(self):
        """Test that WhisperProcessor is used for other languages."""
        # When: Check ASR processor for Japanese
        with patch('video_asr_summary.pipeline.orchestrator.WhisperProcessor') as mock_whisper:
            mock_whisper.return_value = Mock()
            
            processor = self.orchestrator._get_asr_processor('ja')
            
            # Then: Should use WhisperProcessor
            mock_whisper.assert_called_once_with(language="ja")
            assert processor is not None

    def test_should_detect_specialized_asr_correctly(self):
        """Test that _is_using_specialized_asr correctly identifies Chinese vs English."""
        # Given: Mock processors with proper attributes
        specialized_processor = Mock()
        specialized_processor.process_audio = Mock()  # Has process_audio method
        
        # Create a mock without process_audio attribute
        class MockWhisperProcessor:
            def transcribe(self, path):
                pass
        
        whisper_processor = MockWhisperProcessor()
        
        with patch.object(self.orchestrator, '_get_asr_processor') as mock_get_processor:
            # When: Test Chinese language
            mock_get_processor.return_value = specialized_processor
            assert self.orchestrator._is_using_specialized_asr('zh')
            
            # When: Test English language  
            mock_get_processor.return_value = whisper_processor
            assert not self.orchestrator._is_using_specialized_asr('en')

    def test_language_mapping_variations(self):
        """Test that various language code formats map correctly."""
        test_cases = [
            # Chinese variants should use SpecializedASRIntegrator
            ('zh', 'SpecializedASRIntegrator'),
            ('zh-cn', 'SpecializedASRIntegrator'),  
            ('zh-tw', 'SpecializedASRIntegrator'),
            ('chinese', 'SpecializedASRIntegrator'),
            ('mandarin', 'SpecializedASRIntegrator'),
            
            # Other languages should use Whisper
            ('en', 'WhisperProcessor'),
            ('english', 'WhisperProcessor'),
            ('ja', 'WhisperProcessor'),
            ('japanese', 'WhisperProcessor'),
            ('ko', 'WhisperProcessor'),
            ('korean', 'WhisperProcessor'),
            ('fr', 'WhisperProcessor'),
            ('es', 'WhisperProcessor'),
        ]
        
        for language, expected_processor in test_cases:
            with patch('video_asr_summary.pipeline.orchestrator.SpecializedASRIntegrator') as mock_specialized, \
                 patch('video_asr_summary.pipeline.orchestrator.WhisperProcessor') as mock_whisper:
                
                mock_specialized.return_value = Mock()
                mock_whisper.return_value = Mock()
                
                self.orchestrator._get_asr_processor(language)
                
                if expected_processor == 'SpecializedASRIntegrator':
                    mock_specialized.assert_called_once()
                    mock_whisper.assert_not_called()
                else:
                    mock_whisper.assert_called_once()
                    # SpecializedASRIntegrator should not be called for non-Chinese languages
                    mock_specialized.assert_not_called()
