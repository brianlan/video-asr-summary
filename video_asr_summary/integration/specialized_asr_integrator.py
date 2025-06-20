"""Integration of specialized VAD, ASR, punctuation, and diarization processors."""
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from video_asr_summary.core import (
    VADProcessor, VADResult, ASRProcessor, TranscriptionResult,
    PunctuationProcessor, PunctuationResult,
    SpeakerDiarizationProcessor, DiarizationResult,
    EnhancedTranscriptionResult, SpeakerSegment
)
from video_asr_summary.vad.funasr_vad_processor import FunASRVADProcessor
from video_asr_summary.asr.funasr_specialized_processor import FunASRSpecializedProcessor
from video_asr_summary.punctuation.funasr_punc_processor import FunASRPunctuationProcessor
from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor


class SpecializedASRIntegrator:
    """Orchestrator for sequential VAD->ASR->Punctuation->Diarization integration."""

    def __init__(
        self,
        device: str = "auto",
        vad_processor: Optional[VADProcessor] = None,
        asr_processor: Optional[ASRProcessor] = None,
        punctuation_processor: Optional[PunctuationProcessor] = None,
        diarization_processor: Optional[SpeakerDiarizationProcessor] = None,
        diarization_auth_token: Optional[str] = None,
    ) -> None:
        self.device = device
        self._vad_processor = vad_processor or FunASRVADProcessor(device=device)
        self._asr_processor = asr_processor or FunASRSpecializedProcessor(device=device)
        self._punctuation_processor = punctuation_processor or FunASRPunctuationProcessor(device=device)
        # Default diarization processor requires auth token
        if diarization_processor:
            self._diarization_processor = diarization_processor
        else:
            token = diarization_auth_token or ""
            self._diarization_processor = PyannoteAudioProcessor(token, device=device)

    def process_audio(self, audio_path: Path) -> EnhancedTranscriptionResult:
        """Process audio through VAD, ASR, punctuation, and diarization."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Step 1: VAD
        vad_start = time.time()
        vad_result: VADResult = self._vad_processor.detect_voice_activity(audio_path)
        vad_time = time.time() - vad_start

        # If no speech detected, return empty result
        if not vad_result.segments:
            # Empty transcription and speaker attribution
            empty_trans = TranscriptionResult(text="", confidence=0.0, segments=[], language=None, processing_time_seconds=0.0)
            empty_diari = DiarizationResult(segments=[], num_speakers=0, processing_time_seconds=0.0)
            return EnhancedTranscriptionResult(
                transcription=empty_trans,
                diarization=empty_diari,
                speaker_attributed_segments=[],
                processing_time_seconds=vad_time
            )

        # Step 2: ASR on full audio - get both regular and character-level results
        asr_result: TranscriptionResult = self._asr_processor.transcribe(audio_path)
        
        # Get character-level segments for punctuation-aware processing
        try:
            # Try to get character-level segments (FunASR-specific)
            raw_text, character_segments = self._asr_processor.get_character_level_segments(audio_path)  # type: ignore
        except (AttributeError, Exception):
            # Fallback for non-FunASR processors
            raw_text = asr_result.text
            character_segments = asr_result.segments

        # Step 3: Punctuation restoration
        punc_result: PunctuationResult = self._punctuation_processor.restore_punctuation(raw_text)

        # Step 4: NEW APPROACH - Combine character-level timing with punctuation boundaries
        optimized_segments = self._create_punctuation_aware_segments(
            punc_result.text, 
            character_segments,
            raw_text
        )

        # Step 5: Diarization on full audio
        diarization_result: DiarizationResult = self._diarization_processor.diarize(audio_path)

        # Step 6: Attribute speakers to segments
        speaker_segments = self._attribute_speakers_to_segments(optimized_segments, diarization_result.segments)

        # Safely sum processing times (None -> 0)
        asr_time = asr_result.processing_time_seconds or 0.0
        punc_time = punc_result.processing_time_seconds or 0.0
        diar_time = diarization_result.processing_time_seconds or 0.0
        total_asr_punc = asr_time + punc_time
        total_time = vad_time + total_asr_punc + diar_time
        return EnhancedTranscriptionResult(
            transcription=TranscriptionResult(
                text=punc_result.text,
                confidence=punc_result.confidence,
                segments=optimized_segments,
                language=asr_result.language,
                processing_time_seconds=total_asr_punc
            ),
            diarization=diarization_result,
            speaker_attributed_segments=speaker_segments,
            processing_time_seconds=total_time
        )

    def _create_punctuation_aware_segments(
        self,
        punctuated_text: str,
        character_segments: List[Dict[str, Any]],
        original_text: str
    ) -> List[Dict[str, Any]]:
        """Create segments using punctuation boundaries with character-level timing.
        
        This method combines:
        1. Character-level timing precision from FunASR
        2. Natural linguistic boundaries from punctuation model
        
        Args:
            punctuated_text: Text with restored punctuation
            character_segments: Character-level segments from ASR with precise timing
            original_text: Original unpunctuated text for alignment
            
        Returns:
            List of optimized segments with punctuation-based boundaries and precise timing
        """
        if not character_segments or not punctuated_text:
            return []
        
        # Step 1: Find punctuation boundaries in the punctuated text
        punctuation_boundaries = self._find_punctuation_boundaries(punctuated_text)
        
        # Step 2: Map punctuation boundaries to character positions in original text
        char_to_punc_mapping = self._align_original_to_punctuated_text(original_text, punctuated_text)
        
        # Step 3: Create segments based on punctuation boundaries using character timing
        optimized_segments = self._build_segments_from_boundaries(
            punctuation_boundaries,
            char_to_punc_mapping,
            character_segments,
            punctuated_text
        )
        
        return optimized_segments

    def _find_punctuation_boundaries(self, punctuated_text: str) -> List[int]:
        """Find character positions where natural segment boundaries should occur.
        
        Args:
            punctuated_text: Text with punctuation
            
        Returns:
            List of character positions marking segment boundaries
        """
        boundaries = [0]  # Start with beginning of text
        
        # Chinese punctuation marks that indicate natural boundaries
        sentence_endings = ['，', '。', '！', '？', '；', '：']
        
        for i, char in enumerate(punctuated_text):
            if char in sentence_endings:
                # Add position after the punctuation mark
                if i + 1 < len(punctuated_text):
                    boundaries.append(i + 1)
        
        # Always include the end of text
        if boundaries[-1] != len(punctuated_text):
            boundaries.append(len(punctuated_text))
        
        return boundaries

    def _align_original_to_punctuated_text(self, original_text: str, punctuated_text: str) -> Dict[int, int]:
        """Create mapping from original text positions to punctuated text positions.
        
        Args:
            original_text: Text without punctuation (from ASR)
            punctuated_text: Text with restored punctuation
            
        Returns:
            Dict mapping original_pos -> punctuated_pos
        """
        mapping = {}
        orig_idx = 0
        punc_idx = 0
        
        # Skip whitespace and align characters
        while orig_idx < len(original_text) and punc_idx < len(punctuated_text):
            orig_char = original_text[orig_idx]
            punc_char = punctuated_text[punc_idx]
            
            if orig_char == punc_char:
                # Characters match - create mapping
                mapping[orig_idx] = punc_idx
                orig_idx += 1
                punc_idx += 1
            elif punc_char in ['，', '。', '！', '？', '；', '：', ' ']:
                # Punctuation or space in punctuated text - skip
                punc_idx += 1
            elif orig_char == ' ':
                # Space in original - skip
                orig_idx += 1
            else:
                # Characters don't match - advance both (fallback)
                mapping[orig_idx] = punc_idx
                orig_idx += 1
                punc_idx += 1
        
        return mapping

    def _build_segments_from_boundaries(
        self,
        punctuation_boundaries: List[int],
        char_mapping: Dict[int, int],
        character_segments: List[Dict[str, Any]],
        punctuated_text: str
    ) -> List[Dict[str, Any]]:
        """Build final segments using punctuation boundaries and character timing.
        
        Args:
            punctuation_boundaries: Positions in punctuated text where segments should break
            char_mapping: Mapping from original text positions to punctuated text positions  
            character_segments: Character-level segments with timing info
            punctuated_text: Full punctuated text
            
        Returns:
            List of segments with punctuation-based text and character-level timing
        """
        segments = []
        
        for i in range(len(punctuation_boundaries) - 1):
            start_boundary = punctuation_boundaries[i]
            end_boundary = punctuation_boundaries[i + 1]
            
            # Extract text for this segment from punctuated text
            segment_text = punctuated_text[start_boundary:end_boundary].strip()
            
            if not segment_text:
                continue
            
            # Find corresponding character segments for timing
            # This is approximate - we map punctuation boundaries to character segments
            start_char_idx = self._find_closest_character_index(start_boundary, char_mapping, character_segments)
            end_char_idx = self._find_closest_character_index(end_boundary, char_mapping, character_segments)
            
            # Get timing from character segments
            if start_char_idx < len(character_segments) and end_char_idx <= len(character_segments):
                start_time = character_segments[start_char_idx].get("start", 0.0)
                end_time = character_segments[min(end_char_idx - 1, len(character_segments) - 1)].get("end", start_time + 1.0)
                
                # Calculate average confidence from character segments in this range
                confidences = [
                    character_segments[j].get("confidence", 0.9) 
                    for j in range(start_char_idx, min(end_char_idx, len(character_segments)))
                ]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.9
                
                segments.append({
                    "id": len(segments),
                    "start": start_time,
                    "end": end_time,
                    "text": segment_text,
                    "confidence": avg_confidence
                })
        
        return segments

    def _find_closest_character_index(
        self, 
        punctuation_pos: int, 
        char_mapping: Dict[int, int], 
        character_segments: List[Dict[str, Any]]
    ) -> int:
        """Find the character segment index closest to a punctuation boundary position.
        
        Args:
            punctuation_pos: Position in punctuated text
            char_mapping: Mapping from original to punctuated text positions
            character_segments: List of character-level segments
            
        Returns:
            Index in character_segments list
        """
        # Find the original text position that maps closest to punctuation_pos
        best_orig_pos = 0
        min_distance = float('inf')
        
        for orig_pos, punc_pos in char_mapping.items():
            distance = abs(punc_pos - punctuation_pos)
            if distance < min_distance:
                min_distance = distance
                best_orig_pos = orig_pos
        
        # Return the index, ensuring it's within bounds
        return min(best_orig_pos, len(character_segments) - 1)

    def _attribute_speakers_to_segments(
        self,
        transcription_segments: List[Dict[str, Any]],
        diarization_segments: List[SpeakerSegment]
    ) -> List[Dict[str, Any]]:
        """Assign speaker labels to each transcription segment based on overlap."""
        attributed: List[Dict[str, Any]] = []
        for seg in transcription_segments:
            # Ensure numeric start and end times
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or 0.0)
            # Find diarization segment with max overlap
            best_spk = None
            best_overlap = 0.0
            for d in diarization_segments:
                # Compute overlap between segment and diarization turn
                d_start = float(d.start or 0.0)
                d_end = float(d.end or 0.0)
                overlap = max(0.0, min(end, d_end) - max(start, d_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_spk = d
            speaker = best_spk.speaker if best_spk else None
            attributed.append({
                **seg,
                "speaker": speaker,
                "speaker_confidence": best_spk.confidence if best_spk else 0.0
            })
        return attributed
