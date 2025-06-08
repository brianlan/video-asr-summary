# Video ASR Summary Pipeline

A modular video processing pipeline that extracts audio from video files, performs automatic speech recognition (ASR), and generates intelligent summaries.

## Features

- 🎥 Video file processing and metadata extraction
- 🎵 Advanced audio extraction with FFmpeg (speech-optimized settings)
- 📊 Support for chunked processing of long videos
- 🎤 Automatic Speech Recognition (ASR) using state-of-the-art models
- 📝 Intelligent text summarization
- 🧪 Test-driven development with comprehensive test coverage
- 🏗️ Modular architecture with low coupling and high cohesion
- 📦 Easy installation and CLI interface

## Architecture

The pipeline follows a modular design with the following components:

```
video_asr_summary/
├── core/           # Core interfaces and data models
├── video/          # Video processing (OpenCV-based)
├── audio/          # Audio extraction (FFmpeg-based)
├── asr/            # Speech recognition components
├── summarization/  # Text summarization components
├── pipeline/       # Pipeline orchestration
└── cli/            # Command-line interface
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for audio extraction)
# macOS:
brew install ffmpeg
# Ubuntu/Debian:
sudo apt install ffmpeg
```

### Basic Audio Extraction

```python
from video_asr_summary.audio import extract_audio_for_speech_recognition
from pathlib import Path

# Extract audio optimized for speech recognition (16kHz, mono, WAV)
audio_data = extract_audio_for_speech_recognition(
    video_path=Path("input_video.mp4"),
    output_path=Path("speech_audio.wav")
)

print(f"Extracted {audio_data.duration_seconds:.2f}s of audio")
```

### Chunked Processing for Long Videos

```python
from video_asr_summary.audio import AudioExtractorFactory

extractor = AudioExtractorFactory.create_extractor("ffmpeg")

# Process long videos in chunks (useful for ASR)
chunks = extractor.extract_audio_chunks(
    video_path=Path("long_video.mp4"),
    output_dir=Path("audio_chunks/"),
    chunk_duration_seconds=300,  # 5-minute chunks
    sample_rate=16000,           # Speech recognition optimized
    channels=1,                  # Mono
    overlap_seconds=2            # 2-second overlap for continuity
)

print(f"Created {len(chunks)} audio chunks for processing")
```

### Example Demo

Run the included demo script to see audio extraction capabilities:

```bash
python examples/audio_processing_demo.py path/to/your/video.mp4
```

This will demonstrate:
- Speech-optimized audio extraction
- Chunked processing for long videos  
- Custom extraction settings

## Development

This project follows test-driven development principles:

1. **Test-Driven Development**: Write tests first, then implement features
2. **Modular Design**: Keep modules loosely coupled with high cohesion
3. **Incremental Development**: Implement features one by one
4. **Regular Refactoring**: Review and improve code structure regularly
5. **Git Best Practices**: Frequent commits and meaningful commit messages
6. **Requirement Clarification**: Ask questions to ensure correct implementation

### Running Tests
```bash
python run_tests.py  # Uses custom test runner with path setup
# or alternatively:
python -m pytest    # If you prefer direct pytest
```

### Current Status
- ✅ Core data models and interfaces
- ✅ Video processing (OpenCV-based)
- ✅ Audio extraction (FFmpeg-based) with chunked processing support
- 🚧 ASR processing (in development)
- 🚧 Text summarization (planned)
- 🚧 Pipeline orchestration (planned)

### Code Quality
```bash
# Format code
black video_asr_summary/ tests/ examples/
# Sort imports  
isort video_asr_summary/ tests/ examples/
# Lint code
flake8 video_asr_summary/ tests/ examples/
# Type checking
mypy video_asr_summary/
```

## License

MIT License
