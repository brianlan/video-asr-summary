# Video ASR Summary Pipeline

A modular video processing pipeline that extracts audio from video files, performs automatic speech recognition (ASR), and generates intelligent summaries.

## Features

- 🎥 Video file processing and metadata extraction
- 🎵 Advanced audio extraction with FFmpeg (speech-optimized settings)
- 📊 Support for chunked processing of long videos
- 🎤 Automatic Speech Recognition (ASR) using MLX Whisper (local processing)
- 🌐 Multi-language support (English and Chinese with auto-detection)
- ⚡ High-performance speech recognition with confidence scoring
- 🎯 Structured output with timestamps and segment-level confidence
- � Speaker diarization using pyannote.audio (identifies "who spoke when")
- 🔗 Intelligent integration of ASR and diarization results
- �📝 Intelligent text summarization
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
├── diarization/    # Speaker diarization and ASR integration
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

# Note: MLX Whisper is automatically installed with requirements.txt
# and provides fast local speech recognition on Apple Silicon

# For speaker diarization, you'll need a Hugging Face access token:
# 1. Sign up at https://huggingface.co/
# 2. Go to https://huggingface.co/settings/tokens
# 3. Create a new token with read access
# 4. Set the token: export HUGGINGFACE_TOKEN=your_token_here
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

### Speech Recognition

```python
from video_asr_summary.asr import WhisperProcessor
from pathlib import Path

# Initialize ASR processor
processor = WhisperProcessor(language="en")  # or "zh" for Chinese, None for auto-detect

# Transcribe audio file
result = processor.transcribe(Path("speech_audio.wav"))

print(f"Transcribed text: {result.text}")
print(f"Language detected: {result.language}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Processing time: {result.processing_time_seconds:.2f}s")

# Access detailed segments with timestamps
for i, segment in enumerate(result.segments):
    start = segment['start']
    end = segment['end']
    text = segment['text']
    print(f"[{start:.1f}s - {end:.1f}s] {text}")
```

### Complete Video-to-Text Pipeline

```python
from video_asr_summary.audio import FFmpegAudioExtractor
from video_asr_summary.asr import WhisperProcessor
from pathlib import Path
import tempfile

def process_video_to_text(video_path: Path, language: str = 'en'):
    # Extract audio optimized for speech recognition
    extractor = FFmpegAudioExtractor()
    temp_audio = Path(tempfile.mkdtemp()) / 'audio.wav'
    
    audio_data = extractor.extract_audio(
        video_path, temp_audio, 
        sample_rate=16000, channels=1  # Optimal for ASR
    )
    
    # Transcribe audio
    processor = WhisperProcessor(language=language)
    result = processor.transcribe(audio_data.file_path)
    
    # Cleanup
    temp_audio.unlink(missing_ok=True)
    
    return result

# Use the pipeline
transcription = process_video_to_text(Path("video.mp4"), language="en")
print(transcription.text)
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

### Speaker Diarization and Enhanced Transcription

The pipeline supports speaker diarization to identify "who spoke when" in your audio/video:

```python
from video_asr_summary.audio import FFmpegAudioExtractor
from video_asr_summary.asr import WhisperProcessor
from video_asr_summary.diarization import PyannoteAudioProcessor, SegmentBasedIntegrator
from pathlib import Path
import os

def process_video_with_speakers(video_path: Path, hf_token: str):
    # Step 1: Extract audio
    extractor = FFmpegAudioExtractor()
    audio_path = video_path.parent / f"{video_path.stem}_audio.wav"
    audio_data = extractor.extract_audio(video_path, audio_path)
    
    # Step 2: Perform ASR
    whisper = WhisperProcessor()
    transcription = whisper.transcribe(audio_path)
    
    # Step 3: Perform speaker diarization
    diarization_processor = PyannoteAudioProcessor(auth_token=hf_token)
    diarization = diarization_processor.diarize(audio_path)
    
    # Step 4: Integrate ASR and diarization
    integrator = SegmentBasedIntegrator(overlap_threshold=0.5)
    enhanced_result = integrator.integrate(transcription, diarization)
    
    # Print speaker-attributed transcript
    for segment in enhanced_result.speaker_attributed_segments:
        start = segment.get('start', 0.0)
        end = segment.get('end', 0.0)
        text = segment.get('text', '').strip()
        speaker = segment.get('speaker', 'UNKNOWN')
        confidence = segment.get('speaker_confidence', 0.0)
        
        if text:
            print(f"{start:6.1f}-{end:5.1f}s [{speaker}] ({confidence:.2f}): {text}")
    
    # Cleanup
    audio_path.unlink(missing_ok=True)
    return enhanced_result

# Usage (requires Hugging Face token)
hf_token = os.getenv("HUGGINGFACE_TOKEN")
result = process_video_with_speakers(Path("meeting.mp4"), hf_token)
```

This provides:
- Speaker identification and labeling (SPEAKER_00, SPEAKER_01, etc.)
- Confidence scores for speaker assignments
- Perfect alignment between transcript segments and speakers
- Preservation of original ASR results for reference

```

### Example Demos

Run the included demo scripts:

```bash
# Audio extraction demo
python examples/audio_processing_demo.py path/to/your/video.mp4

# ASR processing demo  
python examples/asr_demo.py

# Complete video-to-text pipeline demo
python examples/video_to_text_demo.py

# Speaker diarization demo (requires Hugging Face token)
export HUGGINGFACE_TOKEN=your_token_here
python examples/diarization_demo.py path/to/your/video.mp4
```

These demonstrate:
- Speech-optimized audio extraction
- Chunked processing for long videos  
- Local speech recognition with MLX Whisper
- Confidence scoring and timestamp extraction
- Multi-language support (English/Chinese)
- Speaker diarization with pyannote.audio
- Integration of ASR and diarization results

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
- ✅ ASR processing (MLX Whisper with multi-language support)
- ✅ Speaker diarization (pyannote.audio integration)
- ✅ ASR-diarization integration with confidence scoring
- 🚧 Text summarization (planned)
- 🚧 Pipeline orchestration (planned)
- 🚧 Cross-validation with multiple ASR services (planned)

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

## Console Warnings

During audio processing, you may see various warnings in the console. These are normal and safe to ignore:

- **PyTorch std() warnings** - Numerical stability warnings from pyannote.audio
- **TorchAudio MPEG_LAYER_III warnings** - MP3 metadata compatibility issues  
- **libmpg123 layer3 errors** - Low-level MP3 frame decoding issues

All warnings are handled gracefully by the underlying libraries and do not affect processing quality. For detailed analysis, see [docs/warnings_analysis.md](docs/warnings_analysis.md).

## License

MIT License
