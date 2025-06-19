"""Integration of specialized VAD, ASR, punctuation, and diarization processors."""
import time
import re
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

        # Step 2: ASR on full audio
        asr_result: TranscriptionResult = self._asr_processor.transcribe(audio_path)

        # Step 3: Punctuation restoration
        punc_result: PunctuationResult = self._punctuation_processor.restore_punctuation(asr_result.text)

        # Split punctuated text into segments
        split_segments = self._split_punctuated_text_to_segments(punc_result.text, asr_result.segments)

        # Step 4: Diarization on full audio
        diarization_result: DiarizationResult = self._diarization_processor.diarize(audio_path)

        # Step 5: Attribute speakers to segments
        speaker_segments = self._attribute_speakers_to_segments(split_segments, diarization_result.segments)

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
                segments=split_segments,
                language=asr_result.language,
                processing_time_seconds=total_asr_punc
            ),
            diarization=diarization_result,
            speaker_attributed_segments=speaker_segments,
            processing_time_seconds=total_time
        )

    def _split_punctuated_text_to_segments(
        self,
        punctuated_text: str,
        original_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Split restored punctuation text into the original segments based on punctuation."""
        # Split by sentence-ending punctuation (including comma)
        sentences = re.findall(r'.+?[，。！？；：]|.+$', punctuated_text)
        segments: List[Dict[str, Any]] = []
        for idx, orig in enumerate(original_segments):
            # Default start/end to 0 if missing
            start = orig.get("start") or 0.0
            end = orig.get("end") or 0.0
            text = sentences[idx] if idx < len(sentences) else orig.get("text", "")
            segments.append({
                "id": orig.get("id"),
                "start": start,
                "end": end,
                "text": text,
                "confidence": orig.get("confidence") or 0.0,
            })
        return segments

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
