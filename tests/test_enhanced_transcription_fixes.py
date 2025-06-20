"""Test enhanced transcription result persistence and speaker attribution integration.

Following TDD principle - write failing tests first, then implement fixes.
"""

import json
import tempfile
from pathlib import Path

from video_asr_summary.core import TranscriptionResult, EnhancedTranscriptionResult, DiarizationResult, SpeakerSegment
from video_asr_summary.pipeline.state_manager import StateManager


class TestEnhancedTranscriptionPersistence:
    """Test that EnhancedTranscriptionResult is properly saved and loaded."""
    
    def test_state_manager_should_save_enhanced_transcription(self):
        """Test that StateManager can save EnhancedTranscriptionResult to JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_manager = StateManager(tmp_dir)
            state = state_manager.create_state("/test/video.mp4")
            
            # Create an enhanced transcription result
            basic_transcription = TranscriptionResult(
                text="Hello world, how are you?",
                confidence=0.95,
                segments=[
                    {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world,", "confidence": 0.95},
                    {"id": 1, "start": 2.0, "end": 4.0, "text": "how are you?", "confidence": 0.94}
                ],
                language="en",
                processing_time_seconds=1.5
            )
            
            diarization = DiarizationResult(
                segments=[
                    SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00", confidence=0.95),
                    SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01", confidence=0.93)
                ],
                num_speakers=2,
                processing_time_seconds=0.8
            )
            
            enhanced_transcription = EnhancedTranscriptionResult(
                transcription=basic_transcription,
                diarization=diarization,
                speaker_attributed_segments=[
                    {
                        "id": 0, "start": 0.0, "end": 2.0, 
                        "text": "Hello world,", "confidence": 0.95,
                        "speaker": "SPEAKER_00", "speaker_confidence": 0.95
                    },
                    {
                        "id": 1, "start": 2.0, "end": 4.0,
                        "text": "how are you?", "confidence": 0.94,
                        "speaker": "SPEAKER_01", "speaker_confidence": 0.93
                    }
                ],
                processing_time_seconds=2.3
            )
            
            # This method should exist and save the enhanced transcription
            state_manager.save_enhanced_transcription(state, enhanced_transcription)
            
            # Check that enhanced transcription file was created
            enhanced_file = Path(tmp_dir) / "enhanced_transcription.json"
            assert enhanced_file.exists(), "Enhanced transcription JSON file should be created"
            
            # Verify file contents
            with open(enhanced_file, 'r') as f:
                data = json.load(f)
            
            assert "transcription" in data
            assert "diarization" in data
            assert "speaker_attributed_segments" in data
            assert "processing_time_seconds" in data
            assert "saved_at" in data
            
            # Verify transcription data
            assert data["transcription"]["text"] == "Hello world, how are you?"
            assert len(data["transcription"]["segments"]) == 2
            
            # Verify speaker attributed segments
            assert len(data["speaker_attributed_segments"]) == 2
            assert data["speaker_attributed_segments"][0]["speaker"] == "SPEAKER_00"
            assert data["speaker_attributed_segments"][1]["speaker"] == "SPEAKER_01"
    
    def test_state_manager_should_load_enhanced_transcription(self):
        """Test that StateManager can load EnhancedTranscriptionResult from JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_manager = StateManager(tmp_dir)
            state = state_manager.create_state("/test/video.mp4")
            
            # Create enhanced transcription JSON file manually
            enhanced_file = Path(tmp_dir) / "enhanced_transcription.json"
            enhanced_data = {
                "transcription": {
                    "text": "Hello world, how are you?",
                    "confidence": 0.95,
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world,", "confidence": 0.95},
                        {"id": 1, "start": 2.0, "end": 4.0, "text": "how are you?", "confidence": 0.94}
                    ],
                    "language": "en",
                    "processing_time_seconds": 1.5
                },
                "diarization": {
                    "segments": [
                        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "confidence": 0.95},
                        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01", "confidence": 0.93}
                    ],
                    "num_speakers": 2,
                    "processing_time_seconds": 0.8
                },
                "speaker_attributed_segments": [
                    {
                        "id": 0, "start": 0.0, "end": 2.0,
                        "text": "Hello world,", "confidence": 0.95,
                        "speaker": "SPEAKER_00", "speaker_confidence": 0.95
                    },
                    {
                        "id": 1, "start": 2.0, "end": 4.0,
                        "text": "how are you?", "confidence": 0.94,
                        "speaker": "SPEAKER_01", "speaker_confidence": 0.93
                    }
                ],
                "processing_time_seconds": 2.3,
                "saved_at": "2025-06-20T10:30:00"
            }
            
            with open(enhanced_file, 'w') as f:
                json.dump(enhanced_data, f, indent=2)
            
            # This method should exist and load the enhanced transcription
            enhanced_result = state_manager.load_enhanced_transcription(state)
            
            assert enhanced_result is not None
            assert isinstance(enhanced_result, EnhancedTranscriptionResult)
            assert enhanced_result.transcription.text == "Hello world, how are you?"
            assert len(enhanced_result.speaker_attributed_segments) == 2
            assert enhanced_result.speaker_attributed_segments[0]["speaker"] == "SPEAKER_00"


class TestSpeakerAttributedSegments:
    """Test that transcription segments include speaker information."""
    
    def test_enhanced_transcription_should_merge_speaker_info_into_segments(self):
        """Test that transcription.segments should include speaker information."""
        # Create enhanced transcription with separate speaker attribution
        basic_transcription = TranscriptionResult(
            text="Hello world, how are you?",
            confidence=0.95,
            segments=[
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world,", "confidence": 0.95},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "how are you?", "confidence": 0.94}
            ],
            language="en",
            processing_time_seconds=1.5
        )
        
        diarization = DiarizationResult(
            segments=[
                SpeakerSegment(start=0.0, end=2.0, speaker="SPEAKER_00", confidence=0.95),
                SpeakerSegment(start=2.0, end=4.0, speaker="SPEAKER_01", confidence=0.93)
            ],
            num_speakers=2,
            processing_time_seconds=0.8
        )
        
        speaker_attributed_segments = [
            {
                "id": 0, "start": 0.0, "end": 2.0,
                "text": "Hello world,", "confidence": 0.95,
                "speaker": "SPEAKER_00", "speaker_confidence": 0.95
            },
            {
                "id": 1, "start": 2.0, "end": 4.0,
                "text": "how are you?", "confidence": 0.94,
                "speaker": "SPEAKER_01", "speaker_confidence": 0.93
            }
        ]
        
        enhanced_transcription = EnhancedTranscriptionResult(
            transcription=basic_transcription,
            diarization=diarization,
            speaker_attributed_segments=speaker_attributed_segments,
            processing_time_seconds=2.3
        )
        
        # The enhanced transcription should have a method to get unified segments
        unified_segments = enhanced_transcription.get_speaker_attributed_transcription_segments()
        
        # Verify that transcription segments now include speaker info
        assert len(unified_segments) == 2
        
        # First segment should have speaker info
        seg1 = unified_segments[0]
        assert seg1["text"] == "Hello world,"
        assert seg1["speaker"] == "SPEAKER_00"
        assert seg1["speaker_confidence"] == 0.95
        assert seg1["confidence"] == 0.95
        
        # Second segment should have speaker info
        seg2 = unified_segments[1]
        assert seg2["text"] == "how are you?"
        assert seg2["speaker"] == "SPEAKER_01"
        assert seg2["speaker_confidence"] == 0.93
        assert seg2["confidence"] == 0.94
    
    def test_enhanced_transcription_should_create_speaker_attributed_transcription_result(self):
        """Test that we can get a TranscriptionResult with speaker-attributed segments."""
        enhanced_transcription = EnhancedTranscriptionResult(
            transcription=TranscriptionResult(
                text="Hello world, how are you?",
                confidence=0.95,
                segments=[
                    {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world,", "confidence": 0.95},
                    {"id": 1, "start": 2.0, "end": 4.0, "text": "how are you?", "confidence": 0.94}
                ],
                language="en",
                processing_time_seconds=1.5
            ),
            diarization=DiarizationResult(segments=[], num_speakers=2),
            speaker_attributed_segments=[
                {
                    "id": 0, "start": 0.0, "end": 2.0,
                    "text": "Hello world,", "confidence": 0.95,
                    "speaker": "SPEAKER_00", "speaker_confidence": 0.95
                },
                {
                    "id": 1, "start": 2.0, "end": 4.0,
                    "text": "how are you?", "confidence": 0.94,
                    "speaker": "SPEAKER_01", "speaker_confidence": 0.93
                }
            ],
            processing_time_seconds=2.3
        )
        
        # Should be able to get a TranscriptionResult with speaker-attributed segments
        speaker_transcription = enhanced_transcription.to_speaker_attributed_transcription()
        
        assert isinstance(speaker_transcription, TranscriptionResult)
        assert speaker_transcription.text == "Hello world, how are you?"
        assert len(speaker_transcription.segments) == 2
        
        # Segments should now include speaker information
        assert speaker_transcription.segments[0]["speaker"] == "SPEAKER_00"
        assert speaker_transcription.segments[1]["speaker"] == "SPEAKER_01"
        assert "speaker_confidence" in speaker_transcription.segments[0]
        assert "speaker_confidence" in speaker_transcription.segments[1]
