# FunASR GPU Acceleration Guide

This guide explains how to use GPU acceleration with the FunASR processor on macOS (Apple Silicon).

## Overview

The FunASR processor now supports automatic GPU acceleration using Apple's Metal Performance Shaders (MPS) framework. This provides significant performance improvements on Apple Silicon Macs.

## Performance Benefits

Based on testing with a ~45-second Chinese audio file:

- **CPU processing**: 5.93 seconds
- **MPS GPU processing**: 3.12 seconds
- **Speedup**: 1.90x faster with GPU acceleration

## Usage

### Automatic Device Selection (Recommended)

```python
from video_asr_summary.asr.funasr_processor import FunASRProcessor

# Auto-detect the best available device
processor = FunASRProcessor(
    model_path="iic/SenseVoiceSmall",
    language="auto",
    device="auto"  # Will automatically use GPU if available
)

result = processor.transcribe(audio_path)
```

### Manual Device Selection

```python
# Explicitly use Apple Silicon GPU
processor_gpu = FunASRProcessor(device="mps")

# Force CPU usage
processor_cpu = FunASRProcessor(device="cpu")

# Use CUDA (if available on other systems)
processor_cuda = FunASRProcessor(device="cuda")
```

## Device Priority

When `device="auto"`, the processor automatically selects devices in this priority order:

1. **MPS** - Apple Silicon GPU (macOS with Metal support)
2. **CUDA** - NVIDIA GPU (if available)
3. **CPU** - Fallback option

## System Requirements

### For MPS (Apple Silicon GPU) Support:

- Apple Silicon Mac (M1, M2, M3, etc.)
- macOS 12.3 or later
- PyTorch with MPS support (included in recent versions)

### Verification

You can verify GPU support is working by checking the console output:

```
FunASR will use device: mps
```

## Troubleshooting

### MPS Not Available

If you see `FunASR will use device: cpu` instead of `mps`, check:

1. **Hardware**: Ensure you're using an Apple Silicon Mac
2. **macOS Version**: Requires macOS 12.3+
3. **PyTorch Version**: Update to a recent version that supports MPS

### Performance Issues

- First-time usage may be slower due to model download and compilation
- GPU acceleration is most beneficial for longer audio files
- Very short audio clips may not show significant speedup due to GPU initialization overhead

## Integration with Video Processing Pipeline

The GPU-accelerated FunASR processor can be used directly in the video processing pipeline:

```python
from video_asr_summary.pipeline.orchestrator import VideoPipelineOrchestrator

# Create orchestrator with GPU-accelerated ASR
orchestrator = VideoPipelineOrchestrator(
    asr_processor_type="funasr",
    asr_device="auto"  # Will use GPU automatically
)

# Process video with GPU acceleration
result = orchestrator.process_video(
    video_path="path/to/video.mp4",
    output_dir="output/"
)
```

## Best Practices

1. **Use "auto" device selection** for maximum compatibility
2. **Test performance** with your specific audio content
3. **Monitor memory usage** for very long audio files
4. **Keep PyTorch updated** for latest MPS improvements

## Technical Details

The implementation automatically:

- Detects MPS availability using `torch.backends.mps.is_available()`
- Tests device creation to ensure compatibility
- Falls back to CPU if GPU initialization fails
- Maintains identical transcription quality across all devices
