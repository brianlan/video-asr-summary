"""FunASR-based specialized ASR processor for Chinese speech recognition."""

import logging
import re
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
                "revision": self.model_revision,
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
            
            # Fix Chinese character spacing issue
            text = self._fix_chinese_spacing(raw_text)

            # Convert FunASR segments to our format
            segments = self._convert_segments(asr_result)
            
            # Apply spacing fix to segment texts as well
            for segment in segments:
                segment["text"] = self._fix_chinese_spacing(segment["text"])

            # Estimate confidence (FunASR doesn't provide direct confidence scores)
            confidence = self._estimate_confidence(asr_result, segments)

            # Detect language if auto mode
            detected_language = self._detect_language(text, self.language)

            logger.info(f"ASR transcribed {len(text)} characters with {len(segments)} segments")

            return TranscriptionResult(
                text=text,
                confidence=confidence,
                segments=segments,
                language=detected_language,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            raise Exception(f"FunASR ASR transcription failed: {str(e)}") from e

    def _fix_chinese_spacing(self, text: str) -> str:
        """Fix spacing between Chinese characters that some ASR models add.
        
        Args:
            text: Input text potentially with extra spaces between Chinese characters
            
        Returns:
            Text with fixed Chinese character spacing
        """
        if not text:
            return text
        
        # Pattern to match Chinese character/punctuation followed by space followed by Chinese character/punctuation
        # \u4e00-\u9fff is the Unicode range for Chinese characters
        # Include common Chinese punctuation marks
        chinese_chars = r'[\u4e00-\u9fff]'
        chinese_punct = r'[，。！？；：（）【】]'
        
        # Combine patterns
        chinese_all = f'({chinese_chars}|{chinese_punct})'
        pattern = f'{chinese_all}\\s+{chinese_all}'
        
        # Keep applying the fix until no more matches (handles multiple consecutive Chinese chars)
        prev_text = ""
        while prev_text != text:
            prev_text = text
            text = re.sub(pattern, r'\1\2', text)
        
        return text

    def _convert_segments(self, asr_result: dict) -> list[dict]:
        """Convert FunASR segments to standard format.
        
        Args:
            asr_result: Raw FunASR result
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        # Look for 'timestamp' field (sentence-level)
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
        
        # Fallback: Create a single segment if no timestamps available
        if not segments:
            text = asr_result.get("text", "")
            if text:
                logger.warning("No timestamp information available, creating single segment")
                segments.append({
                    "id": 0,
                    "start": 0.0,
                    "end": len(text) * 0.15,  # Rough estimate based on character count
                    "text": text,
                    "confidence": 0.8,  # Lower confidence for estimated timestamps
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
