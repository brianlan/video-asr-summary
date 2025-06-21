"""Test orchestrator diarization optimization - avoid running diarization twice."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from video_asr_summary.core import (
    AudioData, TranscriptionResult, DiarizationResult, 
    EnhancedTranscriptionResult, SpeakerSegment
)
from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator


class TestOrchestratorDiarizationOptimization:
    """Test orchestrator diarization optimization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.orchestrator = PipelineOrchestrator(str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_orchestrator_should_skip_diarization_when_using_specialized_asr(self):
        """Test that orchestrator skips separate diarization when using SpecializedASRIntegrator."""
        # Given: Mock SpecializedASRIntegrator that returns enhanced result with diarization
        mock_specialized_asr = Mock()
        enhanced_result = EnhancedTranscriptionResult(
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
        mock_specialized_asr.process_audio.return_value = enhanced_result

        # Mock the diarization processor to track if it gets called
        mock_diarization_processor = Mock()
        mock_diarization_processor.diarize.return_value = DiarizationResult(
            segments=[SpeakerSegment(start=0.0, end=4.0, speaker="SPEAKER_01", confidence=0.8)],
            num_speakers=1,
            processing_time_seconds=1.0
        )

        audio_data = AudioData(
            file_path=Path(self.temp_dir / "test_audio.wav"),
            duration_seconds=4.0,
            sample_rate=16000,
            channels=1,
            format="wav"
        )

        # Create pipeline state
        state = self.orchestrator.state_manager.create_state("test_video.mp4", str(self.temp_dir))

        # When: Process with mocked processors
        with patch.object(self.orchestrator, '_get_asr_processor', return_value=mock_specialized_asr), \
             patch.object(self.orchestrator, '_diarization_processor', mock_diarization_processor):
            
            # Track method calls
            with patch.object(self.orchestrator, '_diarize_speakers') as mock_diarize_step, \
                 patch.object(self.orchestrator, '_integrate_diarization') as mock_integrate_step:
                
                # Mock the methods to return appropriate values
                mock_diarize_step.return_value = None  # Should be skipped
                mock_integrate_step.return_value = enhanced_result  # Should return the already-enhanced result
                
                transcription = self.orchestrator._transcribe_audio(state, audio_data)

        # Then: Should use SpecializedASRIntegrator
        assert isinstance(transcription, TranscriptionResult)
        assert transcription.text == "Hello world, this is a test."
        
        # And: Should save enhanced transcription
        enhanced_file = self.temp_dir / state.enhanced_transcription_file
        assert enhanced_file.exists()

        # And: The enhanced result should include diarization already
        loaded_enhanced = self.orchestrator.state_manager.load_enhanced_transcription(state)
        assert loaded_enhanced is not None
        assert loaded_enhanced.diarization.num_speakers == 1
        assert len(loaded_enhanced.speaker_attributed_segments) == 2

    def test_orchestrator_should_run_diarization_when_using_regular_asr(self):
        """Test that orchestrator runs diarization when using regular ASR (backward compatibility)."""
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

        # When: Process with regular ASR
        with patch.object(self.orchestrator, '_get_asr_processor', return_value=mock_regular_asr):
            transcription = self.orchestrator._transcribe_audio(state, audio_data)

        # Then: Should return TranscriptionResult
        assert isinstance(transcription, TranscriptionResult)
        assert transcription.text == "Hello world, this is a test."

        # And: Should NOT save enhanced transcription (since regular ASR doesn't provide it)
        enhanced_file = self.temp_dir / state.enhanced_transcription_file
        assert not enhanced_file.exists()

    def test_diarization_should_only_run_once_in_specialized_asr_integrator(self):
        """Test that diarization runs only once (inside SpecializedASRIntegrator) and not as separate step."""
        # Given: Mock SpecializedASRIntegrator with diarization tracking
        mock_specialized_asr = Mock()
        enhanced_result = EnhancedTranscriptionResult(
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
        mock_specialized_asr.process_audio.return_value = enhanced_result

        # Mock separate diarization processor to track calls
        mock_diarization_processor = Mock()
        mock_diarization_processor.diarize.return_value = DiarizationResult(
            segments=[SpeakerSegment(start=0.0, end=4.0, speaker="SPEAKER_01", confidence=0.8)],
            num_speakers=1,
            processing_time_seconds=1.0
        )

        video_path = Path(self.temp_dir / "test_video.mp4")
        # Create a dummy video file for the test
        video_path.touch()
        
        # When: Process video with SpecializedASRIntegrator
        with patch.object(self.orchestrator, '_get_asr_processor', return_value=mock_specialized_asr), \
             patch.object(self.orchestrator, '_diarization_processor', mock_diarization_processor), \
             patch.object(self.orchestrator, '_extract_video_info', return_value=None), \
             patch.object(self.orchestrator, '_extract_audio', return_value=AudioData(
                 file_path=Path(self.temp_dir / "test_audio.wav"),
                 duration_seconds=4.0,
                 sample_rate=16000,
                 channels=1,
                 format="wav"
             )), \
             patch.object(self.orchestrator, '_analyze_content', return_value=None), \
             patch.object(self.orchestrator, '_finalize_results', return_value={"status": "success"}):
            
            result = self.orchestrator.process_video(video_path, analysis_language="en")

        # Then: SpecializedASRIntegrator should be called once
        mock_specialized_asr.process_audio.assert_called_once()
        
        # And: Separate diarization processor should NOT be called at all
        mock_diarization_processor.diarize.assert_not_called()
        
        # And: Result should be processed successfully
        assert result is not None
