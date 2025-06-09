#!/usr/bin/env python3
"""
Test script showing ASR + simulated diarization integration.
This demonstrates the pipeline without requiring a Hugging Face token.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_asr_summary.asr.whisper_processor import WhisperProcessor
from video_asr_summary.core import DiarizationResult, SpeakerSegment
from video_asr_summary.diarization.integrator import SegmentBasedIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_speaker_diarization(duration_seconds: float, num_speakers: int = 2):
    """
    Create simulated speaker diarization results for demonstration.

    Args:
        duration_seconds: Total audio duration
        num_speakers: Number of speakers to simulate

    Returns:
        DiarizationResult with simulated speaker segments
    """
    segments = []

    # Create alternating speaker segments
    segment_duration = 30.0  # 30-second segments
    current_time = 0.0
    speaker_idx = 0

    while current_time < duration_seconds:
        end_time = min(current_time + segment_duration, duration_seconds)

        # Add some variation to segment lengths
        if speaker_idx % 3 == 0:  # Every 3rd segment is longer
            end_time = min(current_time + segment_duration * 1.5, duration_seconds)
        elif speaker_idx % 5 == 0:  # Every 5th segment is shorter
            end_time = min(current_time + segment_duration * 0.7, duration_seconds)

        speaker_label = f"SPEAKER_{speaker_idx % num_speakers:02d}"

        segment = SpeakerSegment(
            start=current_time,
            end=end_time,
            speaker=speaker_label,
            confidence=0.85 + (speaker_idx % 3) * 0.05,  # Vary confidence
        )
        segments.append(segment)

        current_time = end_time
        speaker_idx += 1

    return DiarizationResult(
        segments=segments,
        num_speakers=num_speakers,
        processing_time_seconds=5.2,  # Simulated processing time
    )


def test_asr_with_simulated_diarization(audio_path: str):
    """
    Test ASR with simulated speaker diarization.

    Args:
        audio_path: Path to the audio file
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return False

    logger.info(f"Testing ASR + simulated diarization with: {audio_path}")
    logger.info(f"File size: {audio_path.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        # Step 1: Perform ASR
        logger.info("=" * 60)
        logger.info("Step 1: Performing ASR (Chinese)")
        logger.info("=" * 60)

        whisper_processor = WhisperProcessor(language="zh")
        transcription = whisper_processor.transcribe(audio_path)

        logger.info("ASR Results:")
        logger.info(f"  Language: {transcription.language}")
        logger.info(f"  Confidence: {transcription.confidence:.2f}")
        logger.info(f"  Processing time: {transcription.processing_time_seconds:.2f}s")
        logger.info(f"  Segments: {len(transcription.segments)}")

        # Calculate total duration from segments
        if transcription.segments:
            total_duration = max(seg.get("end", 0) for seg in transcription.segments)
        else:
            total_duration = 300.0  # Fallback estimate

        logger.info(f"  Audio duration: {total_duration:.1f}s")

        # Step 2: Simulate speaker diarization
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Simulating Speaker Diarization")
        logger.info("=" * 60)
        logger.info("(This simulates what pyannote.audio would produce)")

        diarization = simulate_speaker_diarization(total_duration, num_speakers=2)

        logger.info("Simulated Diarization Results:")
        logger.info(f"  Speakers found: {diarization.num_speakers}")
        logger.info(f"  Segments: {len(diarization.segments)}")
        logger.info(f"  Processing time: {diarization.processing_time_seconds:.2f}s")

        # Show speaker timeline
        logger.info("\nSpeaker timeline (first 10 segments):")
        for i, segment in enumerate(diarization.segments[:10]):
            duration = segment.end - segment.start
            logger.info(
                f"  [{i+1}] {segment.start:.1f}-{segment.end:.1f}s ({duration:.1f}s): {segment.speaker} (conf: {segment.confidence:.2f})"
            )

        if len(diarization.segments) > 10:
            logger.info(f"  ... and {len(diarization.segments) - 10} more segments")

        # Step 3: Integrate ASR and diarization
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Integrating ASR + Diarization")
        logger.info("=" * 60)

        integrator = SegmentBasedIntegrator(overlap_threshold=0.5)
        enhanced_result = integrator.integrate(transcription, diarization)

        logger.info("Integration Results:")
        logger.info(
            f"  Enhanced segments: {len(enhanced_result.speaker_attributed_segments)}"
        )
        logger.info(
            f"  Processing time: {enhanced_result.processing_time_seconds:.2f}s"
        )

        # Show integrated results
        logger.info("\nSpeaker-attributed transcript (first 10 segments with text):")
        shown_count = 0
        for segment in enhanced_result.speaker_attributed_segments:
            text = segment.get("text", "").strip()
            if text and shown_count < 10:
                start = segment.get("start", 0.0)
                end = segment.get("end", 0.0)
                speaker = segment.get("speaker", "UNKNOWN")
                confidence = segment.get("speaker_confidence", 0.0)

                # Truncate long text for display
                display_text = text[:100] + "..." if len(text) > 100 else text
                logger.info(
                    f"  [{shown_count+1}] {start:.1f}-{end:.1f}s [{speaker}] ({confidence:.2f}): {display_text}"
                )
                shown_count += 1

        # Attribution statistics
        attributed_count = sum(
            1
            for seg in enhanced_result.speaker_attributed_segments
            if seg.get("speaker") and seg.get("speaker") != "UNKNOWN"
        )
        total_with_text = sum(
            1
            for seg in enhanced_result.speaker_attributed_segments
            if seg.get("text", "").strip()
        )

        logger.info("\nAttribution statistics:")
        logger.info(f"  Total segments with text: {total_with_text}")
        logger.info(f"  Segments with speaker attribution: {attributed_count}")
        if total_with_text > 0:
            logger.info(
                f"  Attribution rate: {attributed_count/total_with_text*100:.1f}%"
            )

        # Speaker statistics
        speaker_stats = {}
        for segment in enhanced_result.speaker_attributed_segments:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            if text:  # Only count segments with actual text
                if speaker in speaker_stats:
                    speaker_stats[speaker]["count"] += 1
                    speaker_stats[speaker]["chars"] += len(text)
                else:
                    speaker_stats[speaker] = {"count": 1, "chars": len(text)}

        logger.info("\nSpeaker contribution:")
        for speaker in sorted(speaker_stats.keys()):
            stats = speaker_stats[speaker]
            logger.info(
                f"  {speaker}: {stats['count']} segments, {stats['chars']} characters"
            )

        logger.info("\n" + "=" * 60)
        logger.info("🎉 ASR + Simulated Diarization Test Completed!")
        logger.info("=" * 60)
        logger.info("This demonstrates the complete pipeline functionality.")
        logger.info("With a real Hugging Face token, pyannote.audio would provide")
        logger.info("actual speaker diarization instead of simulated results.")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test ASR with simulated diarization")
    parser.add_argument(
        "--audio",
        default="/Users/rlan/Downloads/行为ch1.wav",
        help="Path to audio file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = test_asr_with_simulated_diarization(args.audio)

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
