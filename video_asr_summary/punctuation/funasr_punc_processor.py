"""FunASR-based Punctuation Restoration processor."""

import logging
import time
from typing import Optional, Any

from video_asr_summary.core import PunctuationProcessor, PunctuationResult

logger = logging.getLogger(__name__)


class FunASRPunctuationProcessor(PunctuationProcessor):
    """Punctuation restoration processor using FunASR CT-Transformer model."""

    def __init__(
        self,
        model_path: str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        device: str = "auto",
        model_revision: str = "v2.0.4",
        suppress_warnings: bool = False,
    ) -> None:
        """Initialize FunASR punctuation processor.

        Args:
            model_path: FunASR punctuation model path or name
            device: Device to run on ('auto', 'cpu', 'cuda:0', 'mps', etc.)
            model_revision: Model revision/version to use
            suppress_warnings: Whether to suppress ModelScope warnings
        """
        self.model_path = model_path
        self.model_revision = model_revision
        self.device = self._get_optimal_device(device)
        self.suppress_warnings = suppress_warnings
        self._model: Optional[Any] = None
        
        # Suppress warnings if requested
        if self.suppress_warnings:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning, module="modelscope")
        
        logger.info(f"FunASR Punctuation will use device: {self.device}")
        logger.info(f"FunASR Punctuation model: {self.model_path} (revision: {self.model_revision})")

    def _get_optimal_device(self, device_preference: str) -> str:
        """Get the optimal device for punctuation processing.
        
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
        """Lazy initialization of FunASR punctuation model."""
        if self._model is not None:
            return

        try:
            from funasr import AutoModel
            
            # Punctuation model configuration
            model_config = {
                "model": self.model_path,
                "revision": self.model_revision,
                "device": self.device,
                "disable_update": True,
                "trust_remote_code": False,
                "cache_dir": None,
            }
            
            logger.info(f"Initializing FunASR Punctuation model: {self.model_path} on device: {self.device}")
            
            # Try with remote code disabled first
            try:
                self._model = AutoModel(**model_config)
                logger.info("✓ FunASR Punctuation model initialized successfully (local code)")
            except Exception as e:
                # Fallback: try with remote code if local fails
                logger.warning(f"Local initialization failed, trying with remote code: {e}")
                model_config["trust_remote_code"] = True
                self._model = AutoModel(**model_config)
                logger.info("✓ FunASR Punctuation model initialized successfully (remote code)")
                
        except ImportError as e:
            raise ImportError(
                "FunASR is not installed. Please install it with: pip install funasr"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to initialize FunASR Punctuation model: {str(e)}") from e

    def restore_punctuation(self, text: str) -> PunctuationResult:
        """Restore punctuation in text.
        
        Args:
            text: Input text without proper punctuation
            
        Returns:
            PunctuationResult with restored punctuation
        """
        start_time = time.time()

        # Handle empty text
        if not text.strip():
            return PunctuationResult(
                text="",
                confidence=0.0,
                processing_time_seconds=time.time() - start_time,
            )

        self._initialize_model()
        assert self._model is not None, "Punctuation model failed to initialize"

        try:
            # Call FunASR punctuation model
            result = self._model.generate(
                input=text,
                cache={},
                language="auto",
                use_itn=False,  # Punctuation doesn't need inverse text normalization
                batch_size_s=60,
            )

            processing_time = time.time() - start_time

            # Extract result
            if not result or len(result) == 0:
                logger.warning("No punctuation result returned from FunASR")
                return PunctuationResult(
                    text=text,  # Return original text if processing fails
                    confidence=0.5,  # Medium confidence for fallback
                    processing_time_seconds=processing_time,
                )

            # FunASR punctuation returns text with restored punctuation
            punc_result = result[0]
            restored_text = punc_result.get("text", text)

            # Estimate confidence based on punctuation quality
            confidence = self._estimate_confidence(restored_text)

            logger.info(f"Punctuation restored for {len(text)} -> {len(restored_text)} characters")

            return PunctuationResult(
                text=restored_text,
                confidence=confidence,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            # On error, return original text with low confidence
            logger.error(f"FunASR punctuation processing failed: {str(e)}")
            return PunctuationResult(
                text=text,
                confidence=0.3,  # Low confidence for error case
                processing_time_seconds=time.time() - start_time,
            )

    def _estimate_confidence(self, text: str) -> float:
        """Estimate confidence score for punctuation restoration.
        
        Args:
            text: Text with restored punctuation
            
        Returns:
            Estimated confidence score between 0.0 and 1.0
        """
        if not text:
            return 0.0
        
        # Base confidence for FunASR punctuation
        base_confidence = 0.85
        
        # Check for presence of punctuation marks
        punctuation_chars = "，。！？；：""''（）【】"
        punct_count = sum(1 for char in text if char in punctuation_chars)
        
        if punct_count == 0:
            # No punctuation added, lower confidence
            return 0.6
        
        # Adjust confidence based on punctuation density
        text_length = len(text)
        punct_ratio = punct_count / text_length if text_length > 0 else 0
        
        # Good punctuation ratio is around 0.05-0.15 for Chinese text
        if 0.03 <= punct_ratio <= 0.2:
            base_confidence = min(1.0, base_confidence * 1.1)
        elif punct_ratio > 0.2:
            # Too much punctuation might indicate poor quality
            base_confidence *= 0.9
        
        return base_confidence
