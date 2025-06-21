"""Test orchestrator integration with enhanced transcription."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from video_asr_summary.core import (
    AudioData, TranscriptionResult, DiarizationResult, 
    EnhancedTranscriptionResult, SpeakerSegment
)
from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator


class TestOrchestratorEnhancedTranscription:
    """Test orchestrator enhanced transcription integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.orchestrator = PipelineOrchestrator(str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_orchestrator_should_save_enhanced_transcription_when_using_specialized_asr(self):
        """Test that orchestrator saves enhanced transcription when using SpecializedASRIntegrator."""
        # Given: Mock SpecializedASRIntegrator that returns enhanced result
        mock_specialized_asr = Mock()
        mock_specialized_asr.process_audio.return_value = EnhancedTranscriptionResult(
            transcription=TranscriptionResult(
                text="Hello world, this is a test.",
                confidence=0.95,
                segments=[
                    {"start": 0.0, "end": 2.0, "text": "Hello world,", "confidence": 0.96},
                    {"start": 2.0, "end": 4.0, "text": "this is a test.", "confidence": 0.94}
                ],
                language="en",
                processing_time_seconds=1.5
            ),
            diarization=DiarizationResult(
                segments=[
                    SpeakerSegment(start=0.0, end=4.0, speaker="SPEAKER_00", confidence=0.9)
                ],
                num_speakers=1,
                processing_time_seconds=0.5
            ),
            speaker_attributed_segments=[
                {"start": 0.0, "end": 2.0, "text": "Hello world,", "speaker": "SPEAKER_00", "confidence": 0.96},
                {"start": 2.0, "end": 4.0, "text": "this is a test.", "speaker": "SPEAKER_00", "confidence": 0.94}
            ]
        )

        # Mock hasattr to identify it as specialized ASR
        mock_specialized_asr.process_audio = Mock(return_value=mock_specialized_asr.process_audio.return_value)

        audio_data = AudioData(
            file_path=Path(self.temp_dir / "test_audio.wav"),
            duration_seconds=4.0,
            sample_rate=16000,
            channels=1,
            format="wav"
        )

        # Create pipeline state
        state = self.orchestrator.state_manager.create_state("test_video.mp4", str(self.temp_dir))

        # When: Transcribe audio with specialized ASR
        with patch.object(self.orchestrator, '_get_asr_processor', return_value=mock_specialized_asr):
            result = self.orchestrator._transcribe_audio(state, audio_data)

        # Then: Should return TranscriptionResult
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world, this is a test."
        assert result.confidence == 0.95

        # And: Should save enhanced transcription
        enhanced_file = self.temp_dir / state.enhanced_transcription_file
        assert enhanced_file.exists()

        # And: Should be able to load enhanced transcription
        loaded_enhanced = self.orchestrator.state_manager.load_enhanced_transcription(state)
        assert loaded_enhanced is not None
        assert isinstance(loaded_enhanced, EnhancedTranscriptionResult)
        assert loaded_enhanced.transcription.text == "Hello world, this is a test."
        assert loaded_enhanced.diarization.num_speakers == 1

    def test_orchestrator_should_not_save_enhanced_transcription_when_using_regular_asr(self):
        """Test that orchestrator doesn't save enhanced transcription when using regular ASR."""
        # Given: Mock regular ASR processor
        mock_regular_asr = Mock()
        mock_regular_asr.transcribe.return_value = TranscriptionResult(
            text="Hello world, this is a test.",
            confidence=0.90,
            segments=[
                {"start": 0.0, "end": 4.0, "text": "Hello world, this is a test.", "confidence": 0.90}
            ],
            language="en",
            processing_time_seconds=1.0
        )

        # Don't give it process_audio method to distinguish from SpecializedASRIntegrator
        if hasattr(mock_regular_asr, 'process_audio'):
            delattr(mock_regular_asr, 'process_audio')

        audio_data = AudioData(
            file_path=Path(self.temp_dir / "test_audio.wav"),
            duration_seconds=4.0,
            sample_rate=16000,
            channels=1,
            format="wav"
        )

        # Create pipeline state
        state = self.orchestrator.state_manager.create_state("test_video.mp4", str(self.temp_dir))

        # When: Transcribe audio with regular ASR
        with patch.object(self.orchestrator, '_get_asr_processor', return_value=mock_regular_asr):
            result = self.orchestrator._transcribe_audio(state, audio_data)

        # Then: Should return TranscriptionResult
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world, this is a test."

        # And: Should NOT save enhanced transcription
        enhanced_file = self.temp_dir / state.enhanced_transcription_file
        assert not enhanced_file.exists()

        # And: Should not be able to load enhanced transcription
        loaded_enhanced = self.orchestrator.state_manager.load_enhanced_transcription(state)
        assert loaded_enhanced is None
