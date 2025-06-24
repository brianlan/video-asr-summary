"""FunASR-based specialized ASR processor for Chinese speech recognition."""

import logging
import time
from pathlib import Path
from typing import Optional, Any

from video_asr_summary.core import ASRProcessor, TranscriptionResult

logger = logging.getLogger(__name__)


class FunASRSpecializedProcessor(ASRProcessor):
    """Specialized ASR processor using FunASR Paraformer model with Chinese spacing fixes."""

    def __init__(
        self,
        model_path: str = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        language: str = "auto",
        device: str = "auto",
        model_revision: str = "v2.0.4",
        suppress_warnings: bool = False,
    ) -> None:
        """Initialize FunASR specialized ASR processor.

        Args:
            model_path: FunASR ASR model path or name
            language: Language code ('auto', 'zh', 'en', etc.)
            device: Device to run on ('auto', 'cpu', 'cuda:0', 'mps', etc.)
            model_revision: Model revision/version to use
            suppress_warnings: Whether to suppress ModelScope warnings
        """
        self.model_path = model_path
        self.model_revision = model_revision
        self.language = language
        self.device = self._get_optimal_device(device)
        self.suppress_warnings = suppress_warnings
        self._model: Optional[Any] = None
        
        # Suppress warnings if requested
        if self.suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="modelscope")
        
        logger.info(f"FunASR ASR will use device: {self.device}")
        logger.info(f"FunASR ASR model: {self.model_path} (revision: {self.model_revision})")

    def _get_optimal_device(self, device_preference: str) -> str:
        """Get the optimal device for ASR processing.
        
        Args:
            device_preference: User's device preference
            
        Returns:
            Optimal device string
        """
        if device_preference != "auto":
            return device_preference
        
        try:
            import torch
            
            # Priority order: MPS (Apple Silicon) > CUDA > CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Test MPS device creation to ensure it works
                try:
                    test_device = torch.device('mps')
                    _ = torch.randn(2, 2, device=test_device)
                    return "mps"
                except Exception:
                    pass
            
            if torch.cuda.is_available():
                return "cuda"
            
            return "cpu"
        except ImportError:
            # Fallback if torch is not available
            return "cpu"

    def _initialize_model(self):
        """Lazy initialization of FunASR ASR model."""
        if self._model is not None:
            return

        try:
            from funasr import AutoModel
            
            # ASR model configuration
            model_config = {
                "model": self.model_path,
                "model_revision": self.model_revision,
                "device": self.device,
                "disable_update": True,
                "trust_remote_code": False,
                "cache_dir": None,
            }
            
            logger.info(f"Initializing FunASR ASR model: {self.model_path} on device: {self.device}")
            
            # Try with remote code disabled first
            try:
                self._model = AutoModel(**model_config)
                logger.info("✓ FunASR ASR model initialized successfully (local code)")
            except Exception as e:
                # Fallback: try with remote code if local fails
                logger.warning(f"Local initialization failed, trying with remote code: {e}")
                model_config["trust_remote_code"] = True
                self._model = AutoModel(**model_config)
                logger.info("✓ FunASR ASR model initialized successfully (remote code)")
                
        except ImportError as e:
            raise ImportError(
                "FunASR is not installed. Please install it with: pip install funasr"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to initialize FunASR ASR model: {str(e)}") from e

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio to text using FunASR with proper character-level alignment.

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

        self._initialize_model()
        assert self._model is not None, "ASR model failed to initialize"
        start_time = time.time()

        try:
            # Call FunASR ASR model
            result = self._model.generate(
                input=str(audio_path),
                cache={},
                language=self.language,
                use_itn=True,  # Inverse text normalization for better formatting
                batch_size_s=120,
            )

            processing_time = time.time() - start_time

            # Extract result
            if not result or len(result) == 0:
                raise Exception("No transcription result returned from FunASR ASR")

            # FunASR returns list of results, take the first one
            asr_result = result[0]
            raw_text = asr_result.get("text", "")
            
            # NEW APPROACH: Split raw text by spaces to get character pieces
            char_pieces = self._split_text_to_characters(raw_text)
            
            # Get raw segments from FunASR
            raw_segments = asr_result.get("timestamp", [])
            
            # Map each segment to a character piece (1:1 mapping)
            character_segments = self._map_segments_to_characters(raw_segments, char_pieces)
            
            # Combine character segments using VAD/punctuation info for proper text
            final_text, final_segments = self._combine_character_segments(character_segments)

            # Estimate confidence (FunASR doesn't provide direct confidence scores)
            confidence = self._estimate_confidence(asr_result, final_segments)

            # Detect language if auto mode
            detected_language = self._detect_language(final_text, self.language)

            logger.info(f"ASR transcribed {len(final_text)} characters with {len(final_segments)} segments")

            return TranscriptionResult(
                text=final_text,
                confidence=confidence,
                segments=final_segments,
                language=detected_language,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            raise Exception(f"FunASR ASR transcription failed: {str(e)}") from e

    def _split_text_to_characters(self, raw_text: str) -> list[str]:
        """Split raw spaced text into character pieces.
        
        Args:
            raw_text: Raw text from FunASR (e.g., "你 好 世 界")
            
        Returns:
            List of character pieces (e.g., ["你", "好", "世", "界"])
        """
        if not raw_text:
            return []
        
        # Split by whitespace and filter out empty strings
        pieces = [piece.strip() for piece in raw_text.split() if piece.strip()]
        
        logger.debug(f"Split text into {len(pieces)} character pieces: {pieces}")
        return pieces

    def _map_segments_to_characters(self, raw_segments: list, char_pieces: list[str]) -> list[dict]:
        """Map FunASR segments to character pieces with 1:1 correspondence.
        
        Args:
            raw_segments: Raw timestamp segments from FunASR
            char_pieces: Character pieces split from raw text
            
        Returns:
            List of character-level segments with timing info
        """
        character_segments = []
        
        if not raw_segments or not char_pieces:
            logger.warning("No segments or character pieces available")
            return character_segments
        
        # Ensure we have the same number of segments and characters
        if len(raw_segments) != len(char_pieces):
            logger.warning(
                f"Segment count ({len(raw_segments)}) != character count ({len(char_pieces)}). "
                "This may indicate a timing alignment issue."
            )
            # Take minimum to avoid index errors
            min_count = min(len(raw_segments), len(char_pieces))
            raw_segments = raw_segments[:min_count]
            char_pieces = char_pieces[:min_count]
        
        # Map each segment to its corresponding character
        for i, (segment_info, character) in enumerate(zip(raw_segments, char_pieces)):
            if isinstance(segment_info, list) and len(segment_info) >= 2:
                start_ms, end_ms = segment_info[0], segment_info[1]
                
                character_segments.append({
                    "id": i,
                    "start": start_ms / 1000.0,  # Convert to seconds
                    "end": end_ms / 1000.0,
                    "text": character,  # Single character or token
                    "confidence": 0.9,
                    "is_character_level": True  # Flag for post-processing
                })
        
        logger.info(f"Mapped {len(character_segments)} character-level segments")
        return character_segments

    def _combine_character_segments(self, character_segments: list[dict]) -> tuple[str, list[dict]]:
        """Combine character-level segments into proper text with punctuation.
        
        This is where VAD and punctuation model results would be integrated
        to form natural text segments.
        
        Args:
            character_segments: List of character-level segments
            
        Returns:
            Tuple of (final_text, final_segments)
        """
        if not character_segments:
            return "", []
        
        # For now, implement a simple combination strategy
        # TODO: Integrate with VAD and punctuation models here
        
        final_text = "".join(seg["text"] for seg in character_segments)
        
        # Create larger segments by grouping characters
        # This is a placeholder - should use VAD/punctuation boundaries
        final_segments = []
        current_segment = {
            "start": character_segments[0]["start"],
            "text": "",
            "confidence": 0.0
        }
        
        segment_length_threshold = 10  # Characters per segment
        
        for i, char_seg in enumerate(character_segments):
            current_segment["text"] += char_seg["text"]
            current_segment["confidence"] += char_seg["confidence"]
            
            # Break segment on punctuation or length threshold
            should_break = (
                char_seg["text"] in "，。！？；：" or  # Chinese punctuation
                len(current_segment["text"]) >= segment_length_threshold or
                i == len(character_segments) - 1  # Last character
            )
            
            if should_break:
                current_segment["end"] = char_seg["end"]
                current_segment["confidence"] /= len(current_segment["text"])
                current_segment["id"] = len(final_segments)
                
                final_segments.append(current_segment.copy())
                
                # Start new segment if not the last character
                if i < len(character_segments) - 1:
                    current_segment = {
                        "start": character_segments[i + 1]["start"],
                        "text": "",
                        "confidence": 0.0
                    }
        
        logger.info(f"Combined into {len(final_segments)} final segments")
        return final_text, final_segments

    def _estimate_confidence(self, asr_result: dict, segments: list[dict]) -> float:
        """Estimate confidence score for FunASR results.
        
        Args:
            asr_result: Raw FunASR result
            segments: Converted segments
            
        Returns:
            Estimated confidence score between 0.0 and 1.0
        """
        text = asr_result.get("text", "")
        
        if not text:
            return 0.0
        
        # Base confidence for FunASR (generally high for Chinese)
        base_confidence = 0.9
        
        # Adjust based on text length (very short might be uncertain)
        if len(text) < 10:
            base_confidence *= 0.8
        
        # Adjust based on presence of punctuation (indicates good ASR quality)
        punctuation_chars = "，。！？；：""''（）【】"
        has_punctuation = any(char in text for char in punctuation_chars)
        if has_punctuation:
            base_confidence = min(1.0, base_confidence * 1.05)
        
        return base_confidence

    def _detect_language(self, text: str, language_setting: str) -> Optional[str]:
        """Detect language from transcribed text.
        
        Args:
            text: Transcribed text
            language_setting: Original language setting
            
        Returns:
            Detected language code
        """
        if language_setting != "auto":
            return language_setting
        
        # Simple language detection based on character presence
        if not text:
            return None
        
        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        total_chars = len([char for char in text if char.isalpha() or '\u4e00' <= char <= '\u9fff'])
        
        if total_chars == 0:
            return None
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.5:
            return "zh"
        else:
            return "en"  # Default fallback

    def get_character_level_segments(self, audio_path: Path) -> tuple[str, list[dict]]:
        """Get character-level segments without combining them.
        
        This method is used by the punctuation-aware integrator to get raw 
        character-level timing information.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (raw_text_without_spaces, character_level_segments)
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._initialize_model()
        assert self._model is not None, "ASR model failed to initialize"

        try:
            # Call FunASR ASR model
            result = self._model.generate(
                input=str(audio_path),
                cache={},
                language=self.language,
                use_itn=True,
                batch_size_s=120,
            )

            if not result or len(result) == 0:
                raise Exception("No transcription result returned from FunASR ASR")

            asr_result = result[0]
            raw_text = asr_result.get("text", "")
            
            # Split raw text by spaces to get character pieces
            char_pieces = self._split_text_to_characters(raw_text)
            
            # Get raw segments from FunASR
            raw_segments = asr_result.get("timestamp", [])
            
            # Map each segment to a character piece (1:1 mapping)
            character_segments = self._map_segments_to_characters(raw_segments, char_pieces)
            
            # Return raw text without spaces and character-level segments
            raw_text_no_spaces = "".join(char_pieces)
            
            return raw_text_no_spaces, character_segments

        except Exception as e:
            raise Exception(f"FunASR character-level extraction failed: {str(e)}") from e
