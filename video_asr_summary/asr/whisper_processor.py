"""Whisper-based ASR processor using MLX."""

import time
from pathlib import Path
from typing import Optional, Dict, Any
import mlx_whisper
from video_asr_summary.core import ASRProcessor, TranscriptionResult


class WhisperProcessor(ASRProcessor):
    """ASR processor using MLX Whisper for local speech recognition."""
    
    def __init__(
        self, 
        model_name: str = "mlx-community/whisper-large-v3-turbo",
        language: Optional[str] = None
    ):
        """Initialize WhisperProcessor.
        
        Args:
            model_name: MLX Whisper model name or path
            language: Language hint for recognition (e.g., 'en', 'zh')
        """
        self.model_name = model_name
        self.language = language
    
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio to text using MLX Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResult with transcription, confidence, and segments
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Call MLX Whisper
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_name,
                language=self.language
            )
            
            processing_time = time.time() - start_time
            
            # Extract and validate result components
            text = str(result.get('text', '')) if result.get('text') is not None else ''
            
            raw_segments = result.get('segments', [])
            segments = raw_segments if isinstance(raw_segments, list) else []
            
            raw_language = result.get('language')
            language = str(raw_language) if raw_language is not None and not isinstance(raw_language, list) else None
            
            # Calculate confidence score from segments
            confidence = self._calculate_confidence(segments)
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                segments=segments,
                language=language,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}") from e
    
    def _calculate_confidence(self, segments: list) -> float:
        """Calculate overall confidence score from segments.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not segments:
            return 0.0
        
        # Calculate confidence based on avg_logprob and no_speech_prob
        total_confidence = 0.0
        total_duration = 0.0
        
        for segment in segments:
            duration = segment.get('end', 0) - segment.get('start', 0)
            if duration <= 0:
                continue
                
            # Convert log probability to confidence
            # avg_logprob is typically negative, closer to 0 is better
            avg_logprob = segment.get('avg_logprob', -1.0)
            logprob_confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
            
            # no_speech_prob indicates silence, lower is better for speech
            no_speech_prob = segment.get('no_speech_prob', 0.5)
            speech_confidence = 1.0 - no_speech_prob
            
            # Combine both confidence measures
            segment_confidence = (logprob_confidence + speech_confidence) / 2.0
            
            total_confidence += segment_confidence * duration
            total_duration += duration
        
        return total_confidence / total_duration if total_duration > 0 else 0.0
