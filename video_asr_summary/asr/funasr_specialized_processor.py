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

    def transcribe_chunked(
        self, 
        audio_path: Path, 
        chunk_duration_seconds: float = 30.0,
        chunk_overlap_seconds: float = 1.0
    ) -> TranscriptionResult:
        """Transcribe audio using proper chunked processing to prevent OOM issues.

        This method manually splits the audio into chunks and processes them sequentially,
        which is the correct way to prevent memory issues.

        Args:
            audio_path: Path to audio file
            chunk_duration_seconds: Duration of each chunk in seconds
            chunk_overlap_seconds: Overlap between chunks in seconds

        Returns:
            TranscriptionResult with transcription from all chunks combined

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
            import soundfile as sf
            import numpy as np
            import tempfile
            
            # Load the audio file
            logger.info(f"Loading audio file: {audio_path}")
            speech, sample_rate = sf.read(str(audio_path))
            total_duration = len(speech) / sample_rate
            
            logger.info(f"Audio loaded: {total_duration:.2f}s, {sample_rate}Hz")
            
            # Calculate chunk parameters
            chunk_samples = int(chunk_duration_seconds * sample_rate)
            overlap_samples = int(chunk_overlap_seconds * sample_rate)
            stride_samples = chunk_samples - overlap_samples
            
            all_results = []
            total_processed_time = 0.0
            
            # Process audio in chunks
            chunk_index = 0
            for start_sample in range(0, len(speech), stride_samples):
                end_sample = min(start_sample + chunk_samples, len(speech))
                chunk_audio = speech[start_sample:end_sample]
                
                # Skip very small chunks
                if len(chunk_audio) < sample_rate * 0.5:  # Less than 0.5 seconds
                    break
                
                chunk_start_time = start_sample / sample_rate
                chunk_duration = len(chunk_audio) / sample_rate
                
                logger.info(f"Processing chunk {chunk_index + 1}: {chunk_start_time:.1f}s-{chunk_start_time + chunk_duration:.1f}s ({chunk_duration:.1f}s)")
                
                # Save chunk to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    sf.write(temp_path, chunk_audio, sample_rate)
                
                try:
                    # Process this chunk with FunASR
                    chunk_result = self._model.generate(
                        input=str(temp_path),
                        cache={},
                        language=self.language,
                        use_itn=True,
                        batch_size_s=int(chunk_duration_seconds),
                    )
                    
                    if chunk_result and len(chunk_result) > 0:
                        # Adjust timestamps to account for chunk position in full audio
                        asr_result = chunk_result[0].copy()
                        if 'timestamp' in asr_result and asr_result['timestamp']:
                            adjusted_timestamps = []
                            for timestamp in asr_result['timestamp']:
                                if isinstance(timestamp, list) and len(timestamp) >= 2:
                                    # Convert from ms to seconds and add chunk offset
                                    start_offset = chunk_start_time * 1000  # Convert to ms
                                    adjusted_start = timestamp[0] + start_offset
                                    adjusted_end = timestamp[1] + start_offset
                                    adjusted_timestamps.append([adjusted_start, adjusted_end])
                                else:
                                    adjusted_timestamps.append(timestamp)
                            asr_result['timestamp'] = adjusted_timestamps
                        
                        all_results.append(asr_result)
                        
                finally:
                    # Clean up temporary file
                    temp_path.unlink(missing_ok=True)
                
                chunk_index += 1
            
            processing_time = time.time() - start_time
            
            if not all_results:
                raise Exception("No transcription results returned from chunked processing")
            
            # Combine results from all chunks
            combined_text = ""
            combined_segments = []
            total_confidence = 0.0
            
            for chunk_result in all_results:
                raw_text = chunk_result.get("text", "")
                char_pieces = self._split_text_to_characters(raw_text)
                combined_text += "".join(char_pieces)
                
                # Get segments and keep adjusted timestamps
                raw_segments = chunk_result.get("timestamp", [])
                character_segments = self._map_segments_to_characters(raw_segments, char_pieces)
                
                # Add to combined list with adjusted IDs
                for seg in character_segments:
                    seg["id"] = len(combined_segments)
                    combined_segments.append(seg)
                
                # Accumulate confidence
                if character_segments:
                    total_confidence += sum(seg.get("confidence", 0.9) for seg in character_segments)
            
            # Create final segments by combining character segments
            final_text, final_segments = self._combine_character_segments(combined_segments)
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(combined_segments) if combined_segments else 0.9
            
            # Detect language
            detected_language = self._detect_language(final_text, self.language)
            
            logger.info(f"Chunked ASR completed: {len(final_text)} characters, {len(final_segments)} segments, {len(all_results)} chunks processed")
            
            return TranscriptionResult(
                text=final_text,
                confidence=avg_confidence,
                segments=final_segments,
                language=detected_language,
                processing_time_seconds=processing_time,
            )

        except ImportError as e:
            raise Exception("soundfile library is required for chunked processing. Install with: pip install soundfile") from e
        except Exception as e:
            raise Exception(f"FunASR chunked transcription failed: {str(e)}") from e

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio to text using FunASR with automatic chunked processing for memory efficiency.

        This method automatically detects if chunked processing is needed based on file size
        to prevent out-of-memory issues with large audio files.

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

        # Check file size and audio duration to determine if chunked processing is needed
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        
        # Use chunked processing for files larger than 50MB or by default for safety
        # This prevents the "Invalid buffer size: 43.55 GB" error we saw
        if file_size_mb > 50 or True:  # Always use chunked processing for now
            logger.info(f"Using chunked processing for {file_size_mb:.1f}MB audio file")
            return self.transcribe_chunked(audio_path, chunk_duration_seconds=30.0, chunk_overlap_seconds=1.0)
        
        # Fallback to direct processing for small files (currently disabled for safety)
        return self._transcribe_direct(audio_path)
    
    def _transcribe_direct(self, audio_path: Path) -> TranscriptionResult:
        """Direct transcription without chunking (for small files only).
        
        This is the original implementation that can cause OOM for large files.
        """
        self._initialize_model()
        assert self._model is not None, "ASR model failed to initialize"
        start_time = time.time()

        try:
            # Original direct model call (can cause OOM for large files)
            result = self._model.generate(
                input=str(audio_path),
                cache={},
                language=self.language,
                use_itn=True,  # Inverse text normalization for better formatting
                batch_size_s=60,  # Reduced from 120 but still can cause OOM
            )

            processing_time = time.time() - start_time

            # Extract result
            if not result or len(result) == 0:
                raise Exception("No transcription result returned from FunASR ASR")

            # FunASR returns list of results, take the first one
            asr_result = result[0]
            raw_text = asr_result.get("text", "")
            
            # Split raw text by spaces to get character pieces
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
            raise Exception(f"FunASR direct ASR transcription failed: {str(e)}") from e

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
                batch_size_s=60,
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
