# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
python run_tests.py              # Run all tests with automatic path setup
python run_tests.py -v           # Verbose test output
python run_tests.py tests/test_specific_module.py  # Run specific test file
```

### Code Quality
```bash
# Format code (modifies files)
black video_asr_summary/ tests/ examples/ scripts/
isort video_asr_summary/ tests/ examples/ scripts/

# Check formatting (no modifications)  
python scripts/format_check.py

# Type checking
python scripts/type_check.py
mypy video_asr_summary/

# Linting
flake8 video_asr_summary/ tests/ examples/
```

### Video Processing Pipeline
```bash
# Complete video processing with content analysis
python scripts/process_video.py input.mp4 ./output

# With specific language and content type
python scripts/process_video.py video.mp4 ./output \
  --analysis-language zh --content-type technical_review

# Resume interrupted processing
python scripts/process_video.py video.mp4 ./output --resume

# Check pipeline status
python scripts/process_video.py video.mp4 ./output --status
```

## Architecture Overview

This is a modular video processing pipeline for automatic speech recognition (ASR) and intelligent content analysis. The architecture follows strict separation of concerns with a plugin-based design.

### Core Pipeline Flow
1. **Video Processing** (`video/`) - Extract metadata using OpenCV
2. **Audio Extraction** (`audio/`) - Speech-optimized audio extraction via FFmpeg 
3. **ASR Processing** (`asr/`) - Multi-language transcription with MLX Whisper and FunASR
4. **Speaker Diarization** (`diarization/`) - "Who spoke when" using pyannote.audio
5. **ASR Integration** (`integration/`) - Merge ASR and diarization with confidence scoring
6. **Content Analysis** (`analysis/`) - LLM-powered analysis with structured prompts
7. **Pipeline Orchestration** (`pipeline/`) - State management and resume capabilities

### Key Design Patterns

**Processor Pattern**: Each processing stage implements a consistent interface with `.process()` methods and standardized data models.

**State Management**: Full pipeline state persistence with resume capabilities for long-running operations.

**Multi-Model ASR**: Supports both MLX Whisper (local, Apple Silicon optimized) and FunASR (specialized Chinese processing) with automatic language detection.

**Template-Based Analysis**: Content-specific prompt templates (political_commentary, news_report, technical_review, etc.) with automatic type detection.

### Important Integrations

**MLX Whisper**: Primary ASR engine, requires Apple Silicon for optimal performance. Supports 99 languages with automatic detection.

**FunASR**: Specialized Chinese ASR with advanced punctuation and VAD (Voice Activity Detection). Used for enhanced Chinese language processing.

**pyannote.audio**: Speaker diarization requires Hugging Face authentication token (`HUGGINGFACE_TOKEN` environment variable).

**LLM Analysis**: Supports multiple providers via OpenAI-compatible API. Requires `OPENAI_ACCESS_TOKEN` for content analysis features.

### Testing Strategy

Follows strict TDD approach per Copilot instructions:
- Write failing tests first
- Implement minimal code to pass
- Refactor for maintainability
- Heavy use of mocks for external dependencies (audio files, LLM APIs)
- Integration tests for full pipeline validation

### Package Structure Notes

- `core/`: Shared data models and type definitions (punctuation_types, vad_types)
- `pipeline/orchestrator.py`: Main entry point for complete processing
- `pipeline/state_manager.py`: Handles resume/recovery from interruptions  
- `analysis/prompt_templates.py`: Content-type-specific analysis templates
- `analysis/markdown_converter.py`: Converts analysis results to markdown format

### Environment Requirements

- Python 3.11+
- FFmpeg (audio extraction)
- Apple Silicon recommended for MLX Whisper performance
- Optional: HUGGINGFACE_TOKEN for speaker diarization
- Optional: OPENAI_ACCESS_TOKEN for content analysis

### Common Development Patterns

When adding new processors, implement the standard interface pattern seen in existing processors. All processors should have consistent error handling, logging, and data model return types.

When adding new analysis types, extend `analysis/prompt_templates.py` with new content type templates and update the classifier logic.

The pipeline supports graceful degradation - missing optional components (like LLM analysis) don't break core ASR functionality.