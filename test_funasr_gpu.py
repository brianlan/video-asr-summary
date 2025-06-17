#!/usr/bin/env python3
"""Test FunAS        print("=" * 60)
        print(f"Testing {description} (device='{device}')")
        print("=" * 60)rocessor with GPU acceleration."""

import time
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.insert(0, '/Users/rlan/projects/video-asr-summary')

from video_asr_summary.asr.funasr_processor import FunASRProcessor


def test_funasr_gpu():
    """Test FunASR processor with different device options."""
    audio_path = Path("/Users/rlan/Downloads/ruige-huangjin-4000/audio.wav")
    
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        print("Please ensure the audio file exists at the specified path.")
        return
    
    print(f"Testing FunASR with audio file: {audio_path}")
    print(f"Audio file size: {audio_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Test different device configurations
    devices_to_test = [
        ("auto", "Auto-detect best device"),
        ("cpu", "CPU only"),
        ("mps", "Apple Silicon GPU (MPS)"),
    ]
    
    results = {}
    
    for device, description in devices_to_test:
        print(f"=" * 60)
        print(f"Testing {description} (device='{device}')")
        print("=" * 60)
        
        try:
            # Initialize processor
            processor = FunASRProcessor(
                model_path="iic/SenseVoiceSmall",
                language="auto",
                device=device
            )
            
            # Measure transcription time
            start_time = time.time()
            result = processor.transcribe(audio_path)
            total_time = time.time() - start_time
            
            print(f"Transcription completed in {total_time:.2f} seconds")
            print(f"Processing time (internal): {result.processing_time_seconds:.2f} seconds")
            print(f"Detected language: {result.language}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Number of segments: {len(result.segments)}")
            print(f"Text length: {len(result.text)} characters")
            print()
            print("Transcribed text:")
            print("-" * 40)
            print(result.text[:500])  # Show first 500 characters
            if len(result.text) > 500:
                print(f"... (truncated, full text is {len(result.text)} characters)")
            print()
            
            # Store results for comparison
            results[device] = {
                'total_time': total_time,
                'processing_time': result.processing_time_seconds,
                'text_length': len(result.text),
                'confidence': result.confidence,
                'segments': len(result.segments)
            }
            
        except Exception as e:
            print(f"Error with device '{device}': {e}")
            print()
    
    # Performance comparison
    if len(results) > 1:
        print("=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"{'Device':<10} {'Time (s)':<10} {'Speedup':<10} {'Text Len':<10} {'Confidence':<12}")
        print("-" * 60)
        
        # Use CPU as baseline for speedup calculation
        cpu_time = results.get('cpu', {}).get('total_time', 0)
        
        for device, data in results.items():
            speedup = f"{cpu_time/data['total_time']:.2f}x" if cpu_time > 0 and data['total_time'] > 0 else "N/A"
            print(f"{device:<10} {data['total_time']:<10.2f} {speedup:<10} {data['text_length']:<10} {data['confidence']:<12.3f}")
        
        print()
        if 'mps' in results and 'cpu' in results:
            mps_speedup = cpu_time / results['mps']['total_time']
            print(f"MPS GPU acceleration provides {mps_speedup:.2f}x speedup over CPU!")


if __name__ == "__main__":
    test_funasr_gpu()
