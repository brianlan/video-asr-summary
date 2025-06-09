#!/usr/bin/env python3
"""
Example script demonstrating enhanced audio extraction capabilities.

This script shows how to:
1. Extract audio optimized for speech recognition
2. Extract audio in chunks for processing long videos
3. Handle various video formats and edge cases

Usage:
    python examples/audio_processing_demo.py <video_file>
"""

import sys
from pathlib import Path

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_asr_summary.audio import (
    AudioExtractorFactory,
    extract_audio_for_speech_recognition,
)


def demonstrate_basic_extraction(video_path: Path, output_dir: Path):
    """Demonstrate basic audio extraction optimized for speech recognition."""
    print("1. Basic Speech-Optimized Audio Extraction")
    print("-" * 50)

    output_path = output_dir / "speech_audio.wav"

    try:
        audio_data = extract_audio_for_speech_recognition(
            video_path=video_path, output_path=output_path
        )

        print("✅ Successfully extracted audio:")
        print(f"   Output: {audio_data.file_path}")
        print(f"   Duration: {audio_data.duration_seconds:.2f} seconds")
        print(f"   Sample Rate: {audio_data.sample_rate} Hz")
        print(f"   Channels: {audio_data.channels}")
        print(f"   Format: {audio_data.format}")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")

    print()


def demonstrate_chunked_extraction(video_path: Path, output_dir: Path):
    """Demonstrate chunked audio extraction for long videos."""
    print("2. Chunked Audio Extraction for Long Videos")
    print("-" * 50)

    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    try:
        extractor = AudioExtractorFactory.create_extractor("ffmpeg")

        # Extract in 30-second chunks (small for demo purposes)
        chunks = extractor.extract_audio_chunks(
            video_path=video_path,
            output_dir=chunks_dir,
            chunk_duration_seconds=30.0,
            sample_rate=16000,  # Speech recognition optimized
            channels=1,
            overlap_seconds=2.0,  # 2-second overlap for continuity
        )

        print(f"✅ Successfully created {len(chunks)} audio chunks:")
        for i, chunk in enumerate(chunks):
            print(
                f"   Chunk {i+1}: {chunk.file_path.name} "
                f"({chunk.duration_seconds:.2f}s)"
            )

        total_duration = sum(chunk.duration_seconds for chunk in chunks)
        print(f"   Total audio duration: {total_duration:.2f} seconds")

    except Exception as e:
        print(f"❌ Chunked extraction failed: {e}")

    print()


def demonstrate_custom_extraction(video_path: Path, output_dir: Path):
    """Demonstrate custom audio extraction with specific settings."""
    print("3. Custom Audio Extraction Settings")
    print("-" * 50)

    try:
        extractor = AudioExtractorFactory.create_extractor("ffmpeg")

        # High-quality stereo extraction
        hq_output = output_dir / "high_quality.wav"
        hq_audio = extractor.extract_audio(
            video_path=video_path,
            output_path=hq_output,
            sample_rate=48000,  # High quality
            channels=2,  # Stereo
            format="wav",
        )

        print("✅ High-quality audio extracted:")
        print(f"   Output: {hq_audio.file_path.name}")
        print(f"   Sample Rate: {hq_audio.sample_rate} Hz")
        print(f"   Channels: {hq_audio.channels}")

        # Extract specific time range (first 10 seconds)
        excerpt_output = output_dir / "excerpt.wav"
        excerpt_audio = extractor.extract_audio(
            video_path=video_path,
            output_path=excerpt_output,
            sample_rate=16000,
            channels=1,
            start_time=0.0,
            duration=10.0,
        )

        print("✅ Audio excerpt extracted:")
        print(f"   Output: {excerpt_audio.file_path.name}")
        print(f"   Duration: {excerpt_audio.duration_seconds:.2f} seconds")

    except Exception as e:
        print(f"❌ Custom extraction failed: {e}")

    print()


def main():
    """Main demonstration function."""
    if len(sys.argv) != 2:
        print("Usage: python examples/audio_processing_demo.py <video_file>")
        print("\nThis script demonstrates various audio extraction capabilities.")
        print("The video file should be a valid video format supported by FFmpeg.")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path("audio_output")
    output_dir.mkdir(exist_ok=True)

    print(f"🎥 Processing video: {video_path}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()

    # Run demonstrations
    demonstrate_basic_extraction(video_path, output_dir)
    demonstrate_chunked_extraction(video_path, output_dir)
    demonstrate_custom_extraction(video_path, output_dir)

    print("🎉 Audio processing demonstration complete!")
    print(f"💡 Check the '{output_dir}' directory for extracted audio files.")


if __name__ == "__main__":
    main()
