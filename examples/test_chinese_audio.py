#!/usr/bin/env python3
"""
Test script for Chinese audio processing with Whisper and pyannote.audio.

This script tests both ASR and speaker diarization on a real Chinese audio file.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_asr_summary.asr.whisper_processor import WhisperProcessor
from video_asr_summary.diarization.integrator import SegmentBasedIntegrator
from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_chinese_audio_processing(audio_path: str, hf_token: Optional[str] = None):
    """
    Test ASR and diarization on Chinese audio file.

    Args:
        audio_path: Path to the Chinese audio file
        hf_token: Hugging Face token for pyannote.audio (optional, can be from env)
    """
    audio_path_obj = Path(audio_path)

    if not audio_path_obj.exists():
        logger.error(f"Audio file not found: {audio_path_obj}")
        return False

    logger.info(f"Testing Chinese audio processing with file: {audio_path_obj}")
    logger.info(f"File size: {audio_path_obj.stat().st_size / 1024 / 1024:.2f} MB")

    try:
        # Step 1: Test Whisper ASR with Chinese
        logger.info("=" * 60)
        logger.info("Step 1: Testing Whisper ASR (Chinese)")
        logger.info("=" * 60)

        # Test with language hint for Chinese
        whisper_processor = WhisperProcessor(language="zh")
        transcription = whisper_processor.transcribe(audio_path_obj)

        logger.info("ASR Results:")
        logger.info(f"  Language detected: {transcription.language}")
        logger.info(f"  Confidence: {transcription.confidence:.2f}")
        logger.info(f"  Processing time: {transcription.processing_time_seconds:.2f}s")
        logger.info(f"  Number of segments: {len(transcription.segments)}")
        logger.info(f"  Total text length: {len(transcription.text)} characters")

        # Show first few segments
        logger.info("\nFirst 5 transcript segments:")
        for i, segment in enumerate(transcription.segments[:5]):
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            text = segment.get("text", "").strip()
            logger.info(f"  [{i+1}] {start:.1f}-{end:.1f}s: {text}")

        # Show text preview
        text_preview = (
            transcription.text[:200] + "..."
            if len(transcription.text) > 200
            else transcription.text
        )
        logger.info(f"\nTranscript preview: {text_preview}")

        # Step 2: Test speaker diarization (if token available)
        if not hf_token:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

        if hf_token:
            logger.info("\n" + "=" * 60)
            logger.info("Step 2: Testing Speaker Diarization")
            logger.info("=" * 60)

            diarization_processor = PyannoteAudioProcessor(auth_token=hf_token)
            diarization = diarization_processor.diarize(audio_path_obj)

            logger.info("Diarization Results:")
            logger.info(f"  Speakers found: {diarization.num_speakers}")
            logger.info(f"  Number of segments: {len(diarization.segments)}")
            logger.info(
                f"  Processing time: {diarization.processing_time_seconds:.2f}s"
            )

            # Show speaker timeline
            logger.info("\nSpeaker timeline (first 10 segments):")
            for i, segment in enumerate(diarization.segments[:10]):
                duration = segment.end - segment.start
                logger.info(
                    f"  [{i+1}] {segment.start:.1f}-{segment.end:.1f}s ({duration:.1f}s): {segment.speaker}"
                )

            if len(diarization.segments) > 10:
                logger.info(f"  ... and {len(diarization.segments) - 10} more segments")

            # Calculate speaker statistics
            speaker_stats = {}
            for segment in diarization.segments:
                speaker = segment.speaker
                duration = segment.end - segment.start
                if speaker in speaker_stats:
                    speaker_stats[speaker]["count"] += 1
                    speaker_stats[speaker]["total_time"] += duration
                else:
                    speaker_stats[speaker] = {"count": 1, "total_time": duration}

            logger.info("\nSpeaker statistics:")
            for speaker, stats in sorted(speaker_stats.items()):
                logger.info(
                    f"  {speaker}: {stats['count']} segments, {stats['total_time']:.1f}s total"
                )

            # Step 3: Test integration
            logger.info("\n" + "=" * 60)
            logger.info("Step 3: Testing ASR-Diarization Integration")
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
            logger.info(
                "\nSpeaker-attributed transcript (first 10 segments with text):"
            )
            shown_count = 0
            for i, segment in enumerate(enhanced_result.speaker_attributed_segments):
                text = segment.get("text", "").strip()
                if text and shown_count < 10:  # Only show segments with text
                    start = segment.get("start", 0.0)
                    end = segment.get("end", 0.0)
                    speaker = segment.get("speaker", "UNKNOWN")
                    confidence = segment.get("speaker_confidence", 0.0)

                    logger.info(
                        f"  [{shown_count+1}] {start:.1f}-{end:.1f}s [{speaker}] ({confidence:.2f}): {text}"
                    )
                    shown_count += 1

            # Count attributed vs unattributed segments
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

            logger.info(f"\nAttribution statistics:")
            logger.info(f"  Segments with text: {total_with_text}")
            logger.info(f"  Segments with speaker attribution: {attributed_count}")
            logger.info(
                f"  Attribution rate: {attributed_count/total_with_text*100:.1f}%"
                if total_with_text > 0
                else "N/A"
            )

        else:
            logger.warning("\n" + "=" * 60)
            logger.warning("Step 2: Skipping Speaker Diarization")
            logger.warning("=" * 60)
            logger.warning(
                "No Hugging Face token found. Set HUGGINGFACE_TOKEN environment variable"
            )
            logger.warning("or pass --token argument to test speaker diarization.")
            logger.warning("Get a token at: https://huggingface.co/settings/tokens")

        logger.info("\n" + "=" * 60)
        logger.info("🎉 Testing completed successfully!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Chinese audio processing")
    parser.add_argument(
        "--audio",
        default="/Users/rlan/Downloads/行为ch1.wav",
        help="Path to audio file (default: /Users/rlan/Downloads/行为ch1.wav)",
    )
    parser.add_argument("--token", help="Hugging Face access token")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = test_chinese_audio_processing(args.audio, args.token)

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
