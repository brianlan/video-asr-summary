#!/usr/bin/env python3
"""
Simple import test to verify all diarization modules work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all diarization-related imports work."""
    print("Testing diarization imports...")
    
    try:
        # Core models
        from video_asr_summary.core import (
            SpeakerSegment, 
            DiarizationResult, 
            EnhancedTranscriptionResult,
            SpeakerDiarizationProcessor,
            ASRDiarizationIntegrator
        )
        print("✅ Core diarization models imported successfully")
        
        # Diarization processor
        from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor
        print("✅ PyannoteAudioProcessor imported successfully")
        
        # Integrator
        from video_asr_summary.diarization.integrator import SegmentBasedIntegrator
        print("✅ SegmentBasedIntegrator imported successfully")
        
        # Test data model creation
        speaker_segment = SpeakerSegment(
            start=0.0,
            end=10.0,
            speaker="SPEAKER_00",
            confidence=0.95
        )
        print(f"✅ SpeakerSegment created: {speaker_segment}")
        
        diarization_result = DiarizationResult(
            segments=[speaker_segment],
            num_speakers=1,
            processing_time_seconds=1.5
        )
        print(f"✅ DiarizationResult created: {diarization_result.num_speakers} speakers")
        
        # Test integrator creation
        integrator = SegmentBasedIntegrator(overlap_threshold=0.5)
        print(f"✅ SegmentBasedIntegrator created with threshold: {integrator.overlap_threshold}")
        
        print("\n🎉 All diarization imports and basic functionality working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_pyannote_availability():
    """Test if pyannote.audio is available (optional)."""
    print("\nTesting pyannote.audio availability...")
    
    try:
        import pyannote.audio
        print("✅ pyannote.audio is installed and available")
        return True
    except ImportError:
        print("⚠️  pyannote.audio is not installed (install with: pip install pyannote.audio)")
        return False


if __name__ == "__main__":
    print("=== Diarization Import Test ===")
    
    success = test_imports()
    test_pyannote_availability()
    
    if success:
        print("\n✅ Import test passed!")
        sys.exit(0)
    else:
        print("\n❌ Import test failed!")
        sys.exit(1)
