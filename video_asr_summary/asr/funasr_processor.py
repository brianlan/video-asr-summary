"""FunASR-based ASR processor for Chinese speech recognition."""

import logging
import time
from pathlib import Path
from typing import Optional, Any

from video_asr_summary.core import ASRProcessor, TranscriptionResult

logger = logging.getLogger(__name__)


class FunASRProcessor(ASRProcessor):
    """ASR processor using FunASR for Chinese speech recognition with better punctuation."""

    def __init__(
        self,
        model_path: str = "iic/SenseVoiceSmall",
        language: str = "auto",
        device: str = "auto",
        model_revision: str = "master",  # Added model revision control
        suppress_warnings: bool = False,  # Option to suppress warnings
    ) -> None:
        """Initialize FunASRProcessor.

        Args:
            model_path: FunASR model path or name
            language: Language code ('auto', 'zn', 'en', 'yue', 'ja', 'ko')
            device: Device to run on ('auto', 'cpu', 'cuda:0', 'mps', etc.)
                   'auto' will automatically select the best available device
            model_revision: Model revision/branch to use (default: 'main' for stability)
            suppress_warnings: Whether to suppress ModelScope warnings (default: False)
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
        
        print(f"FunASR will use device: {self.device}")
        print(f"FunASR model: {self.model_path} (revision: {self.model_revision})")

    def _get_optimal_device(self, device_preference: str) -> str:
        """Get the optimal device for FunASR processing.
        
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
        """Lazy initialization of FunASR model."""
        if self._model is not None:
            return

        try:
            from funasr import AutoModel
            
            # Enhanced model configuration to reduce warnings
            model_config = {
                "model": self.model_path,
                "revision": self.model_revision,  # Use specified revision for stability
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 30000},
                "device": self.device,
                "disable_update": True,
                # Reduce warnings by being more explicit about remote code
                "trust_remote_code": False,  # Changed to False to avoid remote code warnings
                # Add cache directory control
                "cache_dir": None,  # Use default cache location
            }
            
            print(f"Initializing FunASR model: {self.model_path} on device: {self.device}")
            
            # Try with remote code disabled first (more secure and stable)
            try:
                self._model = AutoModel(**model_config)
                print("✓ FunASR model initialized successfully (local code)")
            except Exception as e:
                # Fallback: try with remote code if local fails
                print(f"Local initialization failed, trying with remote code: {e}")
                model_config["trust_remote_code"] = True
                self._model = AutoModel(**model_config)
                print("✓ FunASR model initialized successfully (remote code)")
                
        except ImportError as e:
            raise ImportError(
                "FunASR is not installed. Please install it with: pip install funasr"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to initialize FunASR model: {str(e)}") from e

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio to text using FunASR.

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
        assert self._model is not None, "Model failed to initialize"
        start_time = time.time()

        try:
            # Call FunASR with timestamp generation
            result = self._model.generate(
                input=str(audio_path),
                cache={},
                language=self.language,
                use_itn=True,  # Inverse text normalization for better formatting
                batch_size_s=60,
                merge_vad=False,  # Merge voice activity detection segments
                merge_length_s=0,
                # merge_vad=False,  # Don't merge VAD segments to preserve timestamps
                # merge_length_s=0,  # Don't merge segments automatically
                # sentence_timestamp=True,  # Enable sentence-level timestamps
                # word_timestamp=True,  # Enable word-level timestamps if available
            )

            processing_time = time.time() - start_time

            # Extract result
            if not result or len(result) == 0:
                raise Exception("No transcription result returned from FunASR")

            # FunASR returns list of results, take the first one
            asr_result = result[0]
            
            # Apply post-processing for better punctuation
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                text = rich_transcription_postprocess(asr_result.get("text", ""))
            except ImportError:
                # Fallback if postprocess_utils is not available
                text = asr_result.get("text", "")

            # Convert FunASR segments to our format
            segments = self._convert_segments(asr_result)
            
            # Estimate confidence (FunASR doesn't provide direct confidence scores)
            confidence = self._estimate_confidence(asr_result, segments)

            # Detect language if auto mode
            detected_language = self._detect_language(text, self.language)

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                segments=segments,
                language=detected_language,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            raise Exception(f"FunASR transcription failed: {str(e)}") from e

    def _convert_segments(self, asr_result: dict) -> list[dict]:
        """Convert FunASR segments to standard format.
        
        Args:
            asr_result: Raw FunASR result
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        # FunASR may provide timestamp information in different formats
        # Try multiple approaches to extract timestamps
        
        # Approach 1: Look for 'timestamp' field (sentence-level)
        raw_segments = asr_result.get("timestamp", [])
        
        if isinstance(raw_segments, list) and raw_segments:
            logger.info(f"Found {len(raw_segments)} timestamp segments")
            for i, segment_info in enumerate(raw_segments):
                if isinstance(segment_info, list) and len(segment_info) >= 2:
                    # Format: [start_ms, end_ms, text] or [start_ms, end_ms]
                    start_ms, end_ms = segment_info[0], segment_info[1]
                    segment_text = segment_info[2] if len(segment_info) > 2 else ""
                    
                    segments.append({
                        "id": i,
                        "start": start_ms / 1000.0,  # Convert to seconds
                        "end": end_ms / 1000.0,
                        "text": segment_text,
                        "confidence": 0.9,  # Default confidence for FunASR
                    })
        
        # Approach 2: Look for 'sentence_info' field
        if not segments:
            sentence_info = asr_result.get("sentence_info", [])
            if isinstance(sentence_info, list) and sentence_info:
                logger.info(f"Found {len(sentence_info)} sentence info segments")
                for i, sentence in enumerate(sentence_info):
                    if isinstance(sentence, dict):
                        start = sentence.get("start", 0) / 1000.0
                        end = sentence.get("end", start + 1.0) / 1000.0
                        text = sentence.get("text", "")
                        
                        segments.append({
                            "id": i,
                            "start": start,
                            "end": end,
                            "text": text,
                            "confidence": 0.9,
                        })
        
        # Approach 3: Look for word-level timestamps
        if not segments:
            word_info = asr_result.get("word_info", [])
            if isinstance(word_info, list) and word_info:
                logger.info(f"Found {len(word_info)} word-level timestamps")
                # Group words into sentences by looking for punctuation
                current_segment = {"words": [], "start": None, "end": None}
                segment_id = 0
                
                for word in word_info:
                    if isinstance(word, dict):
                        word_text = word.get("text", "")
                        word_start = word.get("start", 0) / 1000.0
                        word_end = word.get("end", word_start + 0.1) / 1000.0
                        
                        if current_segment["start"] is None:
                            current_segment["start"] = word_start
                        
                        current_segment["words"].append(word_text)
                        current_segment["end"] = word_end
                        
                        # End segment on punctuation or every ~15 words
                        if (word_text.endswith((".", "。", "?", "？", "!", "！")) or 
                            len(current_segment["words"]) >= 15):
                            
                            segments.append({
                                "id": segment_id,
                                "start": current_segment["start"],
                                "end": current_segment["end"],
                                "text": "".join(current_segment["words"]),
                                "confidence": 0.9,
                            })
                            
                            current_segment = {"words": [], "start": None, "end": None}
                            segment_id += 1
                
                # Add final segment if it has content
                if current_segment["words"]:
                    segments.append({
                        "id": segment_id,
                        "start": current_segment["start"],
                        "end": current_segment["end"],
                        "text": "".join(current_segment["words"]),
                        "confidence": 0.9,
                    })
        
        # Fallback: Create segments based on text length if no timestamps available
        if not segments:
            text = asr_result.get("text", "")
            if text:
                logger.warning("No timestamp information available, creating segments based on text length")
                # Split text into sentences
                import re
                sentences = re.split(r'[。！？.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if sentences:
                    # Estimate duration per character (Chinese: ~0.3s/char, English: ~0.1s/char)
                    is_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
                    char_duration = 0.3 if is_chinese else 0.1
                    
                    current_time = 0.0
                    for i, sentence in enumerate(sentences):
                        duration = len(sentence) * char_duration
                        segments.append({
                            "id": i,
                            "start": current_time,
                            "end": current_time + duration,
                            "text": sentence,
                            "confidence": 0.8,  # Lower confidence for estimated timestamps
                        })
                        current_time += duration
                else:
                    # Single segment fallback
                    segments.append({
                        "id": 0,
                        "start": 0.0,
                        "end": len(text) * 0.2,  # Rough estimate
                        "text": text,
                        "confidence": 0.7,  # Even lower confidence
                    })
        
        logger.info(f"Converted to {len(segments)} segments")
        return segments

    def _estimate_confidence(self, asr_result: dict, segments: list[dict]) -> float:
        """Estimate confidence score for FunASR results.
        
        Args:
            asr_result: Raw FunASR result
            segments: Converted segments
            
        Returns:
            Estimated confidence score between 0.0 and 1.0
        """
        # FunASR generally produces high-quality results for Chinese
        # We can estimate confidence based on text characteristics
        text = asr_result.get("text", "")
        
        if not text:
            return 0.0
        
        # Base confidence for FunASR (generally high for Chinese)
        base_confidence = 0.9
        
        # Adjust based on text length (very short might be uncertain)
        if len(text) < 10:
            base_confidence *= 0.8
        
        # Adjust based on presence of punctuation (FunASR strength)
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
            # Map FunASR language codes to standard codes
            funasr_to_standard = {
                "zn": "zh",  # Chinese
                "en": "en",  # English
                "yue": "zh-yue",  # Cantonese
                "ja": "ja",  # Japanese
                "ko": "ko",  # Korean
            }
            return funasr_to_standard.get(language_setting, language_setting)
        
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
