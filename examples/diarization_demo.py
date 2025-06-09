#!/usr/bin/env python3
"""
Demo script showing speaker diarization and ASR integration.

This script demonstrates how to:
1. Extract audio from video
2. Perform ASR with Whisper
3. Perform speaker diarization with pyannote.audio
4. Integrate the results to get speaker-attributed transcripts

Note: This demo requires a Hugging Face access token for pyannote.audio.
Get one at: https://huggingface.co/settings/tokens
"""

import os
import sys
from pathlib import Path
import logging
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_asr_summary.audio.extractor import FFmpegAudioExtractor
from video_asr_summary.asr.whisper_processor import WhisperProcessor
from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor
from video_asr_summary.diarization.integrator import SegmentBasedIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_diarization_pipeline(video_path: str, hf_token: Optional[str] = None):
    """
    Demonstrate the complete diarization pipeline.
    
    Args:
        video_path: Path to input video file
        hf_token: Hugging Face access token for pyannote.audio
    """
    video_path_obj = Path(video_path)
    
    if not video_path_obj.exists():
        logger.error(f"Video file not found: {video_path_obj}")
        return
    
    if not hf_token:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.error(
                "Hugging Face token is required for pyannote.audio.\n"
                "Set it via environment variable: export HUGGINGFACE_TOKEN=your_token\n"
                "Or pass it as an argument to this function.\n"
                "Get a token at: https://huggingface.co/settings/tokens"
            )
            return
    
    try:
        # Step 1: Extract audio
        logger.info("Step 1: Extracting audio from video...")
        audio_extractor = FFmpegAudioExtractor()
        audio_path = video_path_obj.parent / f"{video_path_obj.stem}_audio.wav"
        
        audio_data = audio_extractor.extract_audio(video_path_obj, audio_path)
        logger.info(f"Audio extracted: {audio_data}")
        
        # Step 2: Perform ASR
        logger.info("Step 2: Performing speech recognition...")
        whisper_processor = WhisperProcessor()
        transcription = whisper_processor.transcribe(audio_path)
        
        logger.info("Transcription completed:")
        logger.info(f"  Language: {transcription.language}")
        logger.info(f"  Confidence: {transcription.confidence:.2f}")
        logger.info(f"  Duration: {transcription.processing_time_seconds:.2f}s")
        logger.info(f"  Segments: {len(transcription.segments)}")
        logger.info(f"  Text preview: {transcription.text[:200]}...")
        
        # Step 3: Perform speaker diarization
        logger.info("Step 3: Performing speaker diarization...")
        diarization_processor = PyannoteAudioProcessor(auth_token=hf_token)
        diarization = diarization_processor.diarize(audio_path)
        
        logger.info("Diarization completed:")
        logger.info(f"  Speakers found: {diarization.num_speakers}")
        logger.info(f"  Segments: {len(diarization.segments)}")
        logger.info(f"  Duration: {diarization.processing_time_seconds:.2f}s")
        
        # Show speaker timeline
        logger.info("Speaker timeline:")
        for i, segment in enumerate(diarization.segments[:10]):  # Show first 10
            logger.info(f"  {segment.start:.1f}-{segment.end:.1f}s: {segment.speaker}")
        if len(diarization.segments) > 10:
            logger.info(f"  ... and {len(diarization.segments) - 10} more segments")
        
        # Step 4: Integrate ASR and diarization
        logger.info("Step 4: Integrating ASR and diarization results...")
        integrator = SegmentBasedIntegrator(overlap_threshold=0.5)
        enhanced_result = integrator.integrate(transcription, diarization)
        
        logger.info("Integration completed:")
        logger.info(f"  Duration: {enhanced_result.processing_time_seconds:.2f}s")
        logger.info(f"  Enhanced segments: {len(enhanced_result.speaker_attributed_segments)}")
        
        # Step 5: Show results
        logger.info("\nStep 5: Speaker-attributed transcript:")
        logger.info("=" * 60)
        
        for i, segment in enumerate(enhanced_result.speaker_attributed_segments[:15]):  # Show first 15
            start = segment.get('start', 0.0)
            end = segment.get('end', 0.0)
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'UNKNOWN')
            confidence = segment.get('speaker_confidence', 0.0)
            
            if text:  # Only show segments with text
                logger.info(f"{start:6.1f}-{end:5.1f}s [{speaker}] ({confidence:.2f}): {text}")
        
        if len(enhanced_result.speaker_attributed_segments) > 15:
            logger.info(f"... and {len(enhanced_result.speaker_attributed_segments) - 15} more segments")
        
        # Summary statistics
        logger.info("\nSummary Statistics:")
        logger.info("=" * 40)
        
        # Count segments per speaker
        speaker_counts = {}
        speaker_durations = {}
        
        for segment in enhanced_result.speaker_attributed_segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            start = segment.get('start', 0.0)
            end = segment.get('end', 0.0)
            duration = end - start
            
            if speaker in speaker_counts:
                speaker_counts[speaker] += 1
                speaker_durations[speaker] += duration
            else:
                speaker_counts[speaker] = 1
                speaker_durations[speaker] = duration
        
        for speaker in sorted(speaker_counts.keys()):
            count = speaker_counts[speaker]
            duration = speaker_durations[speaker]
            logger.info(f"{speaker}: {count} segments, {duration:.1f}s total")
        
        # Cleanup
        if audio_path.exists():
            audio_path.unlink()
            logger.info(f"Cleaned up temporary audio file: {audio_path}")
        
        logger.info("\nDemo completed successfully!")
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        # Cleanup on error
        if 'audio_path' in locals() and audio_path.exists():
            audio_path.unlink()
        raise


def main():
    """Main function for running the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker diarization demo")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--token", help="Hugging Face access token")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    demo_diarization_pipeline(args.video_path, args.token)


if __name__ == "__main__":
    main()
