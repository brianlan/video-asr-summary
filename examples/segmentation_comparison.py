#!/usr/bin/env python3
"""
Demonstration script showing the difference between Whisper and pyannote segmentation.

This script explains and visualizes:
1. Whisper segmentation (content-based, speech chunks)
2. Pyannote segmentation (speaker-based, who is speaking when)
3. How they integrate to provide speaker-attributed transcripts
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_asr_summary.asr.whisper_processor import WhisperProcessor
from video_asr_summary.diarization.integrator import SegmentBasedIntegrator
from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_separator(title: str):
    """Print a visual separator."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_whisper_segments(segments: list):
    """Print Whisper segments in a readable format."""
    print("\nWHISPER SEGMENTS (Content-Based):")
    print("Format: [start-end] text")
    print("-" * 40)

    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        print(f"{i:2d}. [{start:6.2f}-{end:6.2f}] {text}")


def print_diarization_segments(segments: list):
    """Print speaker diarization segments in a readable format."""
    print("\nPYANNOTE SEGMENTS (Speaker-Based):")
    print("Format: [start-end] Speaker_X")
    print("-" * 40)

    for i, segment in enumerate(segments, 1):
        start = segment.start
        end = segment.end
        speaker = segment.speaker
        print(f"{i:2d}. [{start:6.2f}-{end:6.2f}] {speaker}")


def print_integrated_segments(segments: list):
    """Print integrated segments showing both content and speaker."""
    print("\nINTEGRATED SEGMENTS (Content + Speaker):")
    print("Format: [start-end] Speaker_X: text (confidence)")
    print("-" * 50)

    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        speaker = segment.get("speaker", "Unknown")
        confidence = segment.get("speaker_confidence", 0.0)

        speaker_info = (
            f"{speaker} (conf: {confidence:.2f})" if speaker else "No speaker"
        )
        print(f"{i:2d}. [{start:6.2f}-{end:6.2f}] {speaker_info}: {text}")


def explain_segmentation_differences():
    """Explain the conceptual differences between the two segmentation approaches."""
    print_separator("SEGMENTATION CONCEPT EXPLANATION")

    print(
        """
🎤 WHISPER SEGMENTATION (Content-Based):
   • Purpose: Break audio into meaningful speech chunks for transcription
   • Based on: Natural speech patterns, pauses, sentence boundaries
   • Output: Time-stamped text segments with transcribed content
   • Example: "Hello there" [0.0-1.5], "How are you today?" [2.0-4.2]
   • Focus: WHAT was said and WHEN

🗣️  PYANNOTE SEGMENTATION (Speaker-Based):
   • Purpose: Identify WHO is speaking at any given time
   • Based on: Voice characteristics, speaker changes
   • Output: Time periods labeled with speaker identities
   • Example: Speaker_00 [0.0-3.5], Speaker_01 [3.5-7.2]
   • Focus: WHO was speaking and WHEN

🔗 INTEGRATION (Combined):
   • Purpose: Combine both to get speaker-attributed transcripts
   • Method: Overlap analysis between content and speaker segments
   • Output: Text with speaker attribution
   • Example: "Speaker_00: Hello there" [0.0-1.5], "Speaker_01: How are you?" [2.0-4.2]
   • Focus: WHO said WHAT and WHEN

📊 KEY DIFFERENCES:
   • Whisper segments align with speech content (sentences, phrases)
   • Pyannote segments align with speaker turns (voice changes)
   • They often have different boundaries and don't align perfectly
   • Integration uses overlap analysis to match content to speakers
   • Result: Rich transcripts with both content and speaker information
    """
    )


def demo_with_audio_file(audio_path: Path, hf_token: Optional[str] = None):
    """Demonstrate the segmentation difference with a real audio file."""
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return False

    print_separator(f"PROCESSING: {audio_path.name}")

    try:
        # Initialize processors
        whisper = WhisperProcessor()

        # Step 1: Whisper ASR
        print("🎤 Running Whisper ASR...")
        transcription_result = whisper.transcribe(audio_path)

        print(
            f"✅ Whisper completed in {transcription_result.processing_time_seconds:.2f}s"
        )
        print(f"   Language: {transcription_result.language}")
        print(f"   Confidence: {transcription_result.confidence:.3f}")
        print(f"   Total segments: {len(transcription_result.segments)}")

        # Display Whisper segments
        print_whisper_segments(transcription_result.segments[:10])  # Show first 10

        # Step 2: Speaker Diarization (if token provided)
        if hf_token:
            print("\n🗣️  Running pyannote speaker diarization...")
            try:
                diarizer = PyannoteAudioProcessor(auth_token=hf_token)
                diarization_result = diarizer.diarize(audio_path)

                print(
                    f"✅ Diarization completed in {diarization_result.processing_time_seconds:.2f}s"
                )
                print(f"   Speakers found: {diarization_result.num_speakers}")
                print(f"   Total segments: {len(diarization_result.segments)}")

                # Display diarization segments
                print_diarization_segments(
                    diarization_result.segments[:10]
                )  # Show first 10

                # Step 3: Integration
                print("\n🔗 Integrating ASR and diarization...")
                integrator = SegmentBasedIntegrator(overlap_threshold=0.3)
                integrated_result = integrator.integrate(
                    transcription_result, diarization_result
                )

                print(
                    f"✅ Integration completed in {integrated_result.processing_time_seconds:.2f}s"
                )

                # Display integrated segments
                print_integrated_segments(
                    integrated_result.speaker_attributed_segments[:10]
                )

                # Show alignment analysis
                print_separator("ALIGNMENT ANALYSIS")
                analyze_segment_alignment(
                    transcription_result.segments[:5], diarization_result.segments[:5]
                )

            except Exception as e:
                print(f"❌ Diarization failed: {e}")
                print(
                    "💡 This might be due to missing Hugging Face token or model license issues"
                )
        else:
            print("\n⚠️  Skipping diarization (no Hugging Face token provided)")
            print("   To see full demo, provide HF token as second argument")

        return True

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False


def analyze_segment_alignment(whisper_segments: list, diarization_segments: list):
    """Analyze how Whisper and pyannote segments align."""
    print("Comparing first few segments to show alignment differences:")
    print("\nWHISPER vs PYANNOTE TIMING:")
    print("W = Whisper segment, D = Diarization segment")
    print("-" * 60)

    # Create a simple timeline visualization
    timeline = {}

    # Add Whisper segments
    for i, seg in enumerate(whisper_segments):
        start, end = seg.get("start", 0), seg.get("end", 0)
        timeline[f"W{i+1}"] = (start, end, seg.get("text", "")[:30] + "...")

    # Add diarization segments
    for i, seg in enumerate(diarization_segments):
        start, end = seg.start, seg.end
        timeline[f"D{i+1}"] = (start, end, seg.speaker)

    # Sort by start time
    sorted_segments = sorted(timeline.items(), key=lambda x: x[1][0])

    for label, (start, end, info) in sorted_segments:
        segment_type = "Whisper" if label.startswith("W") else "Diarization"
        print(f"{label:3} [{start:6.2f}-{end:6.2f}] {segment_type:12}: {info}")

    print("\n💡 Notice how:")
    print("   • Whisper segments follow speech content boundaries")
    print("   • Diarization segments follow speaker change boundaries")
    print("   • They rarely align perfectly - that's why integration is needed!")


def main():
    """Main demonstration function."""
    print_separator("WHISPER vs PYANNOTE SEGMENTATION DEMO")

    # First, explain the concepts
    explain_segmentation_differences()

    # Check for command line arguments
    if len(sys.argv) < 2:
        print_separator("USAGE")
        print("To see the demo with a real audio file:")
        print(f"python {sys.argv[0]} <audio_file_path> [huggingface_token]")
        print("\nExample:")
        print(f"python {sys.argv[0]} /path/to/audio.wav hf_xxxxxxxxxxxx")
        print("\nWithout a Hugging Face token, only Whisper analysis will be shown.")
        return

    # Get arguments
    audio_path = Path(sys.argv[1])
    hf_token = sys.argv[2] if len(sys.argv) > 2 else None

    # Run the demo
    success = demo_with_audio_file(audio_path, hf_token)

    if success:
        print_separator("SUMMARY")
        print(
            """
✅ This demo shows how Whisper and pyannote provide complementary segmentation:

1. 🎤 WHISPER gives us WHAT was said (content-based segments)
2. 🗣️  PYANNOTE gives us WHO was speaking (speaker-based segments)  
3. 🔗 INTEGRATION combines them to get WHO said WHAT

The key insight: These are two different types of segmentation that serve
different purposes but work together to create rich, speaker-attributed
transcripts.
        """
        )
    else:
        print("\n❌ Demo could not complete successfully.")


if __name__ == "__main__":
    main()
