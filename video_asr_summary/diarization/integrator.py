"""ASR and diarization integration processor."""

import time
from typing import List
import logging

from ..core import (
    ASRDiarizationIntegrator, 
    TranscriptionResult, 
    DiarizationResult, 
    EnhancedTranscriptionResult,
    SpeakerSegment
)


logger = logging.getLogger(__name__)


class SegmentBasedIntegrator(ASRDiarizationIntegrator):
    """Integrates ASR and diarization results using segment-based alignment."""
    
    def __init__(self, overlap_threshold: float = 0.5):
        """
        Initialize the integrator.
        
        Args:
            overlap_threshold: Minimum overlap ratio to assign speaker to segment
        """
        self.overlap_threshold = overlap_threshold
    
    def integrate(
        self, 
        transcription: TranscriptionResult, 
        diarization: DiarizationResult
    ) -> EnhancedTranscriptionResult:
        """
        Integrate transcription and diarization results.
        
        Args:
            transcription: Result from ASR processor
            diarization: Result from diarization processor
            
        Returns:
            EnhancedTranscriptionResult with speaker attribution
        """
        start_time = time.time()
        
        try:
            # Create speaker-attributed segments
            speaker_attributed_segments = []
            
            for segment in transcription.segments:
                # Extract timing information from segment
                segment_start = segment.get('start', 0.0)
                segment_end = segment.get('end', 0.0)
                
                # Skip segments with invalid timing (start >= end)
                # This can happen with very short segments or transcription errors
                # We preserve the segment but mark it as having no speaker attribution
                if segment_start >= segment_end:
                    logger.warning(f"Invalid segment timing: {segment_start}-{segment_end}")
                    # Add segment without speaker attribution
                    enhanced_segment = segment.copy()
                    enhanced_segment['speaker'] = None
                    enhanced_segment['confidence'] = 0.0
                    speaker_attributed_segments.append(enhanced_segment)
                    continue
                
                # Find best matching speaker
                best_speaker, confidence = self._find_best_speaker(
                    segment_start, segment_end, diarization.segments
                )
                
                # Create enhanced segment
                enhanced_segment = segment.copy()
                enhanced_segment['speaker'] = best_speaker
                enhanced_segment['confidence'] = confidence
                
                speaker_attributed_segments.append(enhanced_segment)
            
            processing_time = time.time() - start_time
            
            result = EnhancedTranscriptionResult(
                transcription=transcription,
                diarization=diarization,
                speaker_attributed_segments=speaker_attributed_segments,
                processing_time_seconds=processing_time
            )
            
            logger.info(
                f"ASR-diarization integration completed in {processing_time:.2f}s, "
                f"attributed {len(speaker_attributed_segments)} segments"
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"ASR-diarization integration failed: {e}") from e
    
    def _find_best_speaker(
        self, 
        segment_start: float, 
        segment_end: float, 
        speaker_segments: List[SpeakerSegment]
    ) -> tuple[str | None, float]:
        """
        Find the best matching speaker for a transcription segment.
        
        Uses temporal overlap analysis to match transcription segments with
        speaker segments. The overlap ratio is calculated as:
        overlap_duration / transcription_segment_duration
        
        Args:
            segment_start: Start time of transcription segment
            segment_end: End time of transcription segment
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            Tuple of (speaker_id, confidence_score) where:
            - speaker_id: ID of best matching speaker, None if no good match
            - confidence_score: Overlap ratio (0.0-1.0), higher = better match
        """
        if not speaker_segments:
            return None, 0.0
        
        segment_duration = segment_end - segment_start
        if segment_duration <= 0:
            return None, 0.0
        
        best_speaker = None
        best_overlap_ratio = 0.0
        
        # Check overlap with each speaker segment
        for speaker_segment in speaker_segments:
            # Calculate temporal overlap between transcription and speaker segments
            overlap_start = max(segment_start, speaker_segment.start)
            overlap_end = min(segment_end, speaker_segment.end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                # Overlap ratio = how much of the transcription segment is covered
                overlap_ratio = overlap_duration / segment_duration
                
                # Track the speaker with the highest overlap ratio
                if overlap_ratio > best_overlap_ratio:
                    best_overlap_ratio = overlap_ratio
                    best_speaker = speaker_segment.speaker
        
        # Only assign speaker if overlap meets threshold
        if best_overlap_ratio >= self.overlap_threshold:
            return best_speaker, best_overlap_ratio
        else:
            return None, best_overlap_ratio
