"""Test suite for Specialized ASR Integration processor."""

from unittest.mock import Mock, patch
from pathlib import Path

from video_asr_summary.core import (
    TranscriptionResult, 
    DiarizationResult, 
    EnhancedTranscriptionResult,
    SpeakerSegment,
    VADResult,
    VADSegment,
    PunctuationResult
)
from video_asr_summary.integration.specialized_asr_integrator import SpecializedASRIntegrator


class TestSpecializedASRIntegrator:
    """Test specialized ASR integration processor."""
    
    def test_initialization_default_params(self):
        """Test integrator initialization with default parameters."""
        integrator = SpecializedASRIntegrator()
        
        assert integrator.device == "auto"
        assert integrator._vad_processor is not None
        assert integrator._asr_processor is not None
        assert integrator._punctuation_processor is not None
        assert integrator._diarization_processor is not None
    
    def test_initialization_custom_device(self):
        """Test integrator initialization with custom device."""
        integrator = SpecializedASRIntegrator(device="mps")
        
        assert integrator.device == "mps"
    
    @patch('pathlib.Path.exists')
    def test_process_audio_file_not_found(self, mock_exists):
        """Test error handling for non-existent audio file."""
        mock_exists.return_value = False
        
        integrator = SpecializedASRIntegrator()
        non_existent_path = Path("/non/existent/audio.wav")
        
        try:
            integrator.process_audio(non_existent_path)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
    
    @patch('pathlib.Path.exists')
    def test_process_audio_success(self, mock_exists):
        """Test successful audio processing through the complete pipeline."""
        mock_exists.return_value = True
        
        # Mock all processors
        mock_vad = Mock()
        mock_asr = Mock()
        mock_punc = Mock()
        mock_diarization = Mock()
        
        # Mock VAD result
        vad_result = VADResult(
            segments=[
                VADSegment(start=0.0, end=2.0, confidence=0.95),
                VADSegment(start=3.0, end=5.0, confidence=0.9)
            ],
            total_speech_duration=4.0,
            processing_time_seconds=0.1
        )
        mock_vad.detect_voice_activity.return_value = vad_result
        
        # Mock ASR result
        asr_result = TranscriptionResult(
            text="你好 世界 这是 测试",  # Spaced Chinese
            confidence=0.9,
            segments=[
                {"id": 0, "start": 0.0, "end": 2.0, "text": "你好 世界", "confidence": 0.9},
                {"id": 1, "start": 3.0, "end": 5.0, "text": "这是 测试", "confidence": 0.9}
            ],
            language="zh",
            processing_time_seconds=1.0
        )
        mock_asr.transcribe.return_value = asr_result
        
        # Mock punctuation result
        punc_result = PunctuationResult(
            text="你好世界，这是测试。",  # Fixed spacing and added punctuation
            confidence=0.95,
            processing_time_seconds=0.2
        )
        mock_punc.restore_punctuation.return_value = punc_result
        
        # Mock diarization result
        diarization_result = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00", confidence=0.9),
                SpeakerSegment(start=3.0, end=5.0, speaker="SPEAKER_01", confidence=0.85)
            ],
            num_speakers=2,
            processing_time_seconds=0.5
        )
        mock_diarization.diarize.return_value = diarization_result
        
        integrator = SpecializedASRIntegrator()
        integrator._vad_processor = mock_vad
        integrator._asr_processor = mock_asr
        integrator._punctuation_processor = mock_punc
        integrator._diarization_processor = mock_diarization
        
        audio_path = Path("/test/audio.wav")
        result = integrator.process_audio(audio_path)
        
        # Verify result structure
        assert isinstance(result, EnhancedTranscriptionResult)
        assert result.transcription.text == "你好世界，这是测试。"  # Punctuation restored
        assert result.transcription.language == "zh"
        assert result.diarization.num_speakers == 2
        assert len(result.speaker_attributed_segments) == 2
        
        # Verify speaker attribution
        segment1 = result.speaker_attributed_segments[0]
        assert segment1["speaker"] == "SPEAKER_00"
        assert segment1["text"] == "你好世界，"
        
        segment2 = result.speaker_attributed_segments[1]
        assert segment2["speaker"] == "SPEAKER_01"
        assert segment2["text"] == "这是测试。"
        
        # Verify all processors were called
        mock_vad.detect_voice_activity.assert_called_once_with(audio_path)
        mock_asr.transcribe.assert_called_once_with(audio_path)
        mock_punc.restore_punctuation.assert_called_once_with("你好 世界 这是 测试")
        mock_diarization.diarize.assert_called_once_with(audio_path)
    
    def test_attribute_speakers_to_segments(self):
        """Test speaker attribution to transcription segments."""
        integrator = SpecializedASRIntegrator()
        
        # Mock transcription segments
        transcription_segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "你好世界，", "confidence": 0.9},
            {"id": 1, "start": 3.0, "end": 5.0, "text": "这是测试。", "confidence": 0.9}
        ]
        
        # Mock diarization segments
        diarization_segments = [
            SpeakerSegment(start=0.0, end=2.5, speaker="SPEAKER_00", confidence=0.9),
            SpeakerSegment(start=2.5, end=5.0, speaker="SPEAKER_01", confidence=0.85)
        ]
        
        result = integrator._attribute_speakers_to_segments(
            transcription_segments, 
            diarization_segments
        )
        
        assert len(result) == 2
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_01"
    
    def test_processing_with_no_speech_detected(self):
        """Test processing when no speech is detected."""
        integrator = SpecializedASRIntegrator()
        
        # Mock empty VAD result
        mock_vad = Mock()
        vad_result = VADResult(
            segments=[],
            total_speech_duration=0.0,
            processing_time_seconds=0.1
        )
        mock_vad.detect_voice_activity.return_value = vad_result
        integrator._vad_processor = mock_vad
        
        # Mock other processors
        mock_asr = Mock()
        mock_punc = Mock()
        mock_diarization = Mock()
        integrator._asr_processor = mock_asr
        integrator._punctuation_processor = mock_punc
        integrator._diarization_processor = mock_diarization
        
        with patch('pathlib.Path.exists', return_value=True):
            audio_path = Path("/test/silent_audio.wav")
            result = integrator.process_audio(audio_path)
            
            # Should still return a result but with empty content
            assert isinstance(result, EnhancedTranscriptionResult)
            assert result.transcription.text == ""
            assert len(result.speaker_attributed_segments) == 0
            
            # VAD should be called, but ASR and others might be skipped
            mock_vad.detect_voice_activity.assert_called_once()
