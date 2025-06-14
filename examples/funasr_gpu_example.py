#!/usr/bin/env python3
"""Example demonstrating FunASR with GPU acceleration."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/Users/rlan/projects/video-asr-summary')

from video_asr_summary.asr.funasr_processor import FunASRProcessor


def main():
    """Demonstrate FunASR GPU usage."""
    
    # Example 1: Auto-detect best device (recommended)
    print("=== Example 1: Auto-detect best device ===")
    processor_auto = FunASRProcessor(
        model_path="iic/SenseVoiceSmall",
        language="auto",  # Auto-detect language
        device="auto"     # Auto-detect best device (GPU if available)
    )
    
    # Example 2: Explicitly use Apple Silicon GPU
    print("\n=== Example 2: Explicitly use Apple Silicon GPU ===")
    processor_mps = FunASRProcessor(
        model_path="iic/SenseVoiceSmall", 
        language="zn",    # Chinese
        device="mps"      # Apple Silicon GPU
    )
    
    # Example 3: Force CPU usage (for comparison)
    print("\n=== Example 3: Force CPU usage ===")
    processor_cpu = FunASRProcessor(
        model_path="iic/SenseVoiceSmall",
        language="auto",
        device="cpu"      # CPU only
    )
    
    # Test with your audio file
    audio_path = Path("/Users/rlan/Downloads/ruige-huangjin-4000/audio.wav")
    
    if audio_path.exists():
        print(f"\nTranscribing: {audio_path}")
        
        # Use the auto-detect processor (recommended)
        result = processor_auto.transcribe(audio_path)
        
        print(f"Processing time: {result.processing_time_seconds:.2f} seconds")
        print(f"Detected language: {result.language}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Text: {result.text[:100]}...")
        
    else:
        print(f"Audio file not found: {audio_path}")
        print("Please update the path to point to your audio file.")


if __name__ == "__main__":
    main()
