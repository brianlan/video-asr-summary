# Video ASR Summary Pipeline

A modular video processing pipeline that extracts audio from video files, performs automatic speech recognition (ASR), and generates intelligent summaries.

## Features

- 🎥 Video file processing and audio extraction
- 🎤 Automatic Speech Recognition (ASR) using state-of-the-art models
- 📝 Intelligent text summarization
- 🧪 Test-driven development with comprehensive test coverage
- 🏗️ Modular architecture with low coupling and high cohesion
- 📦 Easy installation and CLI interface

## Architecture

The pipeline follows a modular design with the following components:

```
video_asr_summary/
├── core/           # Core interfaces and base classes
├── video/          # Video processing components
├── audio/          # Audio extraction and processing
├── asr/            # Speech recognition components
├── summarization/  # Text summarization components
├── pipeline/       # Pipeline orchestration
└── cli/            # Command-line interface
```

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## Usage

### CLI
```bash
video-asr-summary process --input video.mp4 --output summary.txt
```

### Python API
```python
from video_asr_summary import VideoASRPipeline

pipeline = VideoASRPipeline()
result = pipeline.process("video.mp4")
print(result.summary)
```

## Development

This project follows test-driven development principles:

1. Write tests first
2. Implement minimal code to pass tests
3. Refactor and improve
4. Keep modules loosely coupled

### Running Tests
```bash
python run_tests.py  # Uses custom test runner with path setup
# or alternatively:
python -m pytest    # If you prefer direct pytest
```

### Code Quality
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## License

MIT License
