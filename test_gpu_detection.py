#!/usr/bin/env python3
"""Test script to detect available GPU acceleration options for FunASR on macOS."""

import sys
import torch
import platform

def check_gpu_availability():
    """Check what GPU acceleration options are available."""
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print()
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # Check MPS (Metal Performance Shaders) availability - macOS GPU acceleration
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
        if torch.backends.mps.is_available():
            print("MPS device can be used for GPU acceleration on Apple Silicon!")
        else:
            print("MPS not available - this might be an Intel Mac or older macOS")
    else:
        print("MPS backend not available in this PyTorch version")
    print()
    
    # Test device creation
    devices_to_test = ['cpu']
    
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices_to_test.append('mps')
    
    print("Testing device creation:")
    for device_name in devices_to_test:
        try:
            device = torch.device(device_name)
            # Test tensor creation on device
            _ = torch.randn(10, 10, device=device)
            print(f"✓ {device_name}: Device creation and tensor allocation successful")
        except Exception as e:
            print(f"✗ {device_name}: Failed - {e}")
    
    print()
    return devices_to_test

if __name__ == "__main__":
    available_devices = check_gpu_availability()
    
    print("Recommended device priority for FunASR:")
    if 'mps' in available_devices:
        print("1. 'mps' - Apple Silicon GPU (recommended for macOS)")
    if 'cuda' in available_devices:
        print("2. 'cuda' - NVIDIA GPU")
    print("3. 'cpu' - CPU fallback")
