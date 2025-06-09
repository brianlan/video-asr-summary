#!/usr/bin/env python3
"""
Quick test script to verify Apple Silicon GPU (MPS) acceleration for pyannote.audio diarization.
Uses a smaller audio file for faster iteration.
"""

import os
import sys
import time
from pathlib import Path

import torch

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mps_availability():
    """Check if MPS is available."""
    print("🔍 MPS AVAILABILITY CHECK")
    print("=" * 40)

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) is ready!")
        return True
    else:
        print("❌ MPS not available")
        return False


def quick_diarization_test():
    """Test diarization with MPS on small audio file."""
    print("\n🚀 QUICK DIARIZATION TEST")
    print("=" * 40)

    # Check audio file
    audio_file = Path("/Users/rlan/Downloads/audio_000.wav")
    if not audio_file.exists():
        print("❌ Audio file not found:", audio_file)
        return

    # Check HF token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not set")
        return

    print(f"Audio file: {audio_file}")
    print(f"File size: {audio_file.stat().st_size / 1024 / 1024:.1f} MB")

    try:
        from video_asr_summary.diarization import PyannoteAudioProcessor

        # Test with auto device selection (should pick MPS on Apple Silicon)
        print("\n📊 Testing with device='auto'...")
        processor = PyannoteAudioProcessor(auth_token=hf_token, device="auto")
        print(f"Selected device: {processor.device}")

        start_time = time.time()
        result = processor.diarize(audio_file)
        end_time = time.time()

        processing_time = end_time - start_time

        print("✅ Diarization successful!")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Speakers found: {result.num_speakers}")
        print(f"   Segments: {len(result.segments)}")

        # Show first few segments
        print("\n📝 First few segments:")
        for i, segment in enumerate(result.segments[:3]):
            print(
                f"   {i+1}. {segment.start:.1f}-{segment.end:.1f}s: Speaker {segment.speaker}"
            )

        if torch.backends.mps.is_available():
            print("\n🎯 Testing explicit MPS device...")
            processor_mps = PyannoteAudioProcessor(auth_token=hf_token, device="mps")
            start_time = time.time()
            _ = processor_mps.diarize(audio_file)
            end_time = time.time()
            mps_time = end_time - start_time
            print(f"   MPS processing time: {mps_time:.2f}s")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("🍎 QUICK MPS DIARIZATION TEST")
    print("=" * 50)

    if test_mps_availability():
        quick_diarization_test()

    print("\n🎉 Test complete!")
