#!/usr/bin/env python3
"""
Simple example of using pyannote.audio with Apple Silicon GPU (MPS) acceleration.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Run diarization with MPS acceleration."""
    
    # Check for required environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ Please set HUGGINGFACE_TOKEN environment variable")
        print("   export HUGGINGFACE_TOKEN=your_token_here")
        return
    
    # Audio file to process
    audio_file = Path("/Users/rlan/Downloads/audio_000.wav")
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print("🍎 Using Apple Silicon GPU (MPS) for Speaker Diarization")
    print("=" * 60)
    
    try:
        from video_asr_summary.diarization import PyannoteAudioProcessor
        
        # Create processor with automatic device selection
        # On Apple Silicon, this will automatically use MPS for GPU acceleration
        processor = PyannoteAudioProcessor(auth_token=hf_token, device="auto")
        
        print(f"📂 Processing: {audio_file.name}")
        print(f"🖥️  Device: {processor.device}")
        
        # Run diarization
        result = processor.diarize(audio_file)
        
        print("\n✅ Results:")
        print(f"   🔊 Speakers found: {result.num_speakers}")
        print(f"   📊 Segments: {len(result.segments)}")
        print(f"   ⏱️  Processing time: {result.processing_time_seconds:.2f}s")
        
        print("\n📝 Speaker Timeline:")
        for i, segment in enumerate(result.segments):
            print(f"   {i+1:2d}. {segment.start:5.1f}-{segment.end:5.1f}s: {segment.speaker}")
        
        print(f"\n🚀 GPU acceleration enabled! Your M1 Mac processed this in {result.processing_time_seconds:.1f} seconds.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
