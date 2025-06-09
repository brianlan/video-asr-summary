#!/usr/bin/env python3
"""
Test script to verify Apple Silicon GPU (MPS) acceleration for pyannote.audio diarization.

This script tests different device configurations and measures performance.
"""

import os
import time
from pathlib import Path

import torch


def test_device_availability():
    """Test what devices are available."""
    print("🔍 DEVICE AVAILABILITY TEST")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")
    print("CPU available: True")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) is ready for acceleration!")
    else:
        print("❌ MPS not available")


def test_mps_diarization():
    """Test diarization with MPS acceleration."""
    print("\n🚀 TESTING MPS ACCELERATION")
    print("=" * 50)

    # Check if we have the test audio file
    audio_file = Path("/Users/rlan/Downloads/audio_000.wav")
    if not audio_file.exists():
        print("❌ Test audio file not found. Please ensure the audio file exists.")
        return

    # Get HF token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not set. Please set your token.")
        return

    # Test different device configurations
    devices = ["cpu", "mps", "auto"]

    for device in devices:
        print(f"\n📊 Testing with device: {device}")
        print("-" * 30)

        try:
            from video_asr_summary.diarization import PyannoteAudioProcessor

            # Create processor with specified device
            processor = PyannoteAudioProcessor(auth_token=hf_token, device=device)

            # Time the diarization
            start_time = time.time()
            result = processor.diarize(audio_file)
            end_time = time.time()

            processing_time = end_time - start_time

            print("✅ Success!")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Speakers found: {result.num_speakers}")
            print(f"   Segments: {len(result.segments)}")
            print(f"   Device used: {processor.device}")

        except Exception as e:
            print(f"❌ Failed: {e}")


def test_device_selection():
    """Test automatic device selection."""
    print("\n🎯 TESTING DEVICE SELECTION")
    print("=" * 50)

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not set.")
        return

    from video_asr_summary.diarization import PyannoteAudioProcessor

    # Test auto device selection
    processor = PyannoteAudioProcessor(auth_token=hf_token, device="auto")
    print(f"Auto-selected device: {processor.device}")

    # Test manual device selections
    test_devices = ["cpu", "mps", "cuda"]
    for device in test_devices:
        try:
            proc = PyannoteAudioProcessor(auth_token=hf_token, device=device)
            print(f"Manual selection '{device}' -> {proc.device}")
        except Exception as e:
            print(f"Failed to create processor with device '{device}': {e}")


def benchmark_comparison():
    """Compare CPU vs MPS performance if both are available."""
    print("\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 50)

    audio_file = Path("/Users/rlan/Downloads/audio_000.wav")
    if not audio_file.exists():
        print("❌ Test audio file not found.")
        return

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("❌ HUGGINGFACE_TOKEN not set.")
        return

    if not torch.backends.mps.is_available():
        print("❌ MPS not available for benchmarking.")
        return

    from video_asr_summary.diarization import PyannoteAudioProcessor

    devices_to_test = ["cpu", "mps"]
    results = {}

    for device in devices_to_test:
        print(f"\n🔄 Benchmarking {device.upper()}...")

        try:
            processor = PyannoteAudioProcessor(auth_token=hf_token, device=device)

            # Warm up run (models load on first use)
            _ = processor.diarize(audio_file)

            # Timed run
            start_time = time.time()
            processor.diarize(audio_file)
            end_time = time.time()

            processing_time = end_time - start_time
            results[device] = processing_time

            print(f"   ✅ {device.upper()}: {processing_time:.2f}s")

        except Exception as e:
            print(f"   ❌ {device.upper()} failed: {e}")

    # Show comparison
    if len(results) > 1:
        cpu_time = results.get("cpu", 0)
        mps_time = results.get("mps", 0)

        if cpu_time > 0 and mps_time > 0:
            speedup = cpu_time / mps_time
            print("\n🏆 PERFORMANCE SUMMARY:")
            print(f"   CPU time: {cpu_time:.2f}s")
            print(f"   MPS time: {mps_time:.2f}s")
            print(f"   Speedup: {speedup:.2f}x {'🚀' if speedup > 1.1 else '📊'}")


if __name__ == "__main__":
    print("🍎 APPLE SILICON (MPS) DIARIZATION TEST")
    print("=" * 60)

    test_device_availability()
    test_device_selection()
    test_mps_diarization()
    benchmark_comparison()

    print("\n🎉 Testing complete!")
    print("\nTo use MPS acceleration in your code:")
    print("processor = PyannoteAudioProcessor(auth_token=token, device='auto')")
    print("# or explicitly:")
    print("processor = PyannoteAudioProcessor(auth_token=token, device='mps')")
