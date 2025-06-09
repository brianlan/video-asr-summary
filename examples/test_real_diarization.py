#!/usr/bin/env python3
"""
Test real pyannote.audio diarization on Chinese audio file.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor


def test_real_diarization():
    """Test real diarization on Chinese audio file."""
    print("🎤 Testing Real Diarization")
    print("=" * 50)
    
    # Initialize processor
    auth_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not auth_token:
        print("❌ No Hugging Face token found in HUGGINGFACE_TOKEN or HF_TOKEN")
        return False
    
    try:
        processor = PyannoteAudioProcessor(auth_token=auth_token)
        print("✅ PyannoteAudioProcessor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return False
    
    # Test audio file
    audio_file = "/Users/rlan/Downloads/行为ch1.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    print(f"📁 Audio file: {audio_file}")
    print(f"📊 File size: {os.path.getsize(audio_file) / 1024 / 1024:.2f} MB")
    
    # Perform diarization
    print("\n🔄 Starting diarization...")
    print("⏱️  This may take a few minutes on CPU...")
    
    try:
        result = processor.diarize(Path(audio_file))
        
        print("\n✅ Diarization completed successfully!")
        print(f"🎯 Found {result.num_speakers} speakers")
        print(f"📍 Generated {len(result.segments)} segments")
        
        # Get unique speakers from segments
        speakers = set(s.speaker for s in result.segments)
        
        # Show speaker summary
        print("\n📊 Speaker Summary:")
        for speaker_id in sorted(speakers):
            speaker_segments = [s for s in result.segments if s.speaker == speaker_id]
            total_duration = sum(s.end - s.start for s in speaker_segments)
            print(f"   {speaker_id}: {len(speaker_segments)} segments, {total_duration:.1f}s total")
        
        # Show first few segments
        print("\n📝 First 10 segments:")
        for i, segment in enumerate(result.segments[:10]):
            print(f"   [{segment.start:6.1f}s - {segment.end:6.1f}s] {segment.speaker} (conf: {segment.confidence:.3f})")
        
        if len(result.segments) > 10:
            print(f"   ... and {len(result.segments) - 10} more segments")
        
        return True
        
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_diarization()
    
    if success:
        print("\n🎉 Real diarization test completed successfully!")
    else:
        print("\n❌ Real diarization test failed!")
        sys.exit(1)
