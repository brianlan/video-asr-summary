"""
ASR Processing Demo using WhisperProcessor

This example demonstrates how to use the WhisperProcessor for speech recognition.
Note: This is a demonstration - you'll need a real audio file to test with actual transcription.
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_asr_summary.asr import WhisperProcessor
from video_asr_summary.core import TranscriptionResult


def demo_whisper_processor():
    """Demo of WhisperProcessor functionality."""
    print("=== WhisperProcessor Demo ===\n")

    # Initialize processor with default settings (English/Chinese auto-detection)
    print("1. Creating WhisperProcessor with default settings...")
    processor = WhisperProcessor()
    print(f"   Model: {processor.model_name}")
    print(f"   Language: {processor.language or 'Auto-detect'}")
    print()

    # Initialize processor with Chinese language hint
    print("2. Creating WhisperProcessor with Chinese language hint...")
    processor_zh = WhisperProcessor(language="zh")
    print(f"   Model: {processor_zh.model_name}")
    print(f"   Language: {processor_zh.language}")
    print()

    # Initialize processor with smaller model
    print("3. Creating WhisperProcessor with smaller model...")
    processor_fast = WhisperProcessor(
        model_name="mlx-community/whisper-base", language="en"
    )
    print(f"   Model: {processor_fast.model_name}")
    print(f"   Language: {processor_fast.language}")
    print()

    # Create a dummy audio file for testing (you'd replace this with a real file)
    dummy_audio_path = Path("/tmp/dummy_audio.wav")

    print("4. Testing file existence check...")
    try:
        processor.transcribe(dummy_audio_path)
        print("   This shouldn't happen - file doesn't exist!")
    except FileNotFoundError as e:
        print(f"   ✓ Correctly detected missing file: {e}")
    print()

    print("5. Example of what a successful transcription result would look like:")

    # Simulate what a real result would look like
    example_result = TranscriptionResult(
        text="Hello, this is a sample transcription with multiple speakers discussing the video content.",
        confidence=0.85,
        segments=[
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello, this is a sample",
                "avg_logprob": -0.3,
                "no_speech_prob": 0.05,
            },
            {
                "start": 2.5,
                "end": 5.8,
                "text": " transcription with multiple speakers",
                "avg_logprob": -0.4,
                "no_speech_prob": 0.02,
            },
            {
                "start": 5.8,
                "end": 8.2,
                "text": " discussing the video content.",
                "avg_logprob": -0.2,
                "no_speech_prob": 0.01,
            },
        ],
        language="en",
        processing_time_seconds=3.45,
    )

    print(f"   Text: {example_result.text}")
    print(f"   Language: {example_result.language}")
    print(f"   Confidence: {example_result.confidence:.2f}")
    print(f"   Processing time: {example_result.processing_time_seconds:.2f} seconds")
    print(f"   Number of segments: {len(example_result.segments)}")
    print()
    print("   Segments with timestamps:")
    for i, segment in enumerate(example_result.segments):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        confidence = 1.0 - segment["no_speech_prob"]  # Simplified confidence
        print(
            f"     {i+1}. [{start:.1f}s - {end:.1f}s] {text} (confidence: {confidence:.2f})"
        )
    print()

    print("6. To use with a real audio file, you would do:")
    print("   ```python")
    print("   processor = WhisperProcessor(language='en')  # or 'zh' for Chinese")
    print("   result = processor.transcribe(Path('/path/to/your/audio.wav'))")
    print("   print(f'Transcribed: {result.text}')")
    print("   ```")
    print()

    print("=== Demo Complete ===")
    print("\nNext steps:")
    print("- Extract audio from a video file using the audio extractor")
    print("- Use this ASR processor to transcribe the audio")
    print("- Later: implement speaker diarization and cross-validation")


if __name__ == "__main__":
    demo_whisper_processor()
