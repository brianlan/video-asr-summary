"""FunASR-based Voice Activity Detection processor."""

import logging
import time
from pathlib import Path
from typing import Optional, Any

from video_asr_summary.core import VADProcessor, VADResult, VADSegment

logger = logging.getLogger(__name__)


class FunASRVADProcessor(VADProcessor):
    """Voice Activity Detection processor using FunASR FSMN-VAD model."""

    def __init__(
        self,
        model_path: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        device: str = "auto",
        model_revision: str = "v2.0.4",
        suppress_warnings: bool = False,
    ) -> None:
        """Initialize FunASR VAD processor.

        Args:
            model_path: FunASR VAD model path or name
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
        
        logger.info(f"FunASR VAD will use device: {self.device}")
        logger.info(f"FunASR VAD model: {self.model_path} (revision: {self.model_revision})")

    def _get_optimal_device(self, device_preference: str) -> str:
        """Get the optimal device for VAD processing.
        
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
        """Lazy initialization of FunASR VAD model."""
        if self._model is not None:
            return

        try:
            from funasr import AutoModel
            
            # VAD model configuration
            model_config = {
                "model": self.model_path,
                "revision": self.model_revision,
                "device": self.device,
                "disable_update": True,
                "trust_remote_code": False,
                "cache_dir": None,
            }
            
            logger.info(f"Initializing FunASR VAD model: {self.model_path} on device: {self.device}")
            
            # Try with remote code disabled first
            try:
                self._model = AutoModel(**model_config)
                logger.info("✓ FunASR VAD model initialized successfully (local code)")
            except Exception as e:
                # Fallback: try with remote code if local fails
                logger.warning(f"Local initialization failed, trying with remote code: {e}")
                model_config["trust_remote_code"] = True
                self._model = AutoModel(**model_config)
                logger.info("✓ FunASR VAD model initialized successfully (remote code)")
                
        except ImportError as e:
            raise ImportError(
                "FunASR is not installed. Please install it with: pip install funasr"
            ) from e
        except Exception as e:
            raise Exception(f"Failed to initialize FunASR VAD model: {str(e)}") from e

    def detect_voice_activity(self, audio_path: Path) -> VADResult:
        """Detect voice activity in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            VADResult with detected speech segments
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If VAD processing fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._initialize_model()
        assert self._model is not None, "VAD model failed to initialize"
        start_time = time.time()

        try:
            # Call FunASR VAD model
            result = self._model.generate(
                input=str(audio_path),
                cache={},
                language="auto",
                use_itn=False,  # VAD doesn't need inverse text normalization
                batch_size_s=60,
            )

            processing_time = time.time() - start_time

            # Extract VAD segments
            if not result or len(result) == 0:
                logger.warning("No VAD result returned from FunASR")
                return VADResult(
                    segments=[],
                    total_speech_duration=0.0,
                    processing_time_seconds=processing_time,
                )

            # FunASR VAD returns segments in 'value' field
            vad_result = result[0]
            raw_segments = vad_result.get("value", [])
            
            # Convert raw segments to VADSegment objects
            segments = []
            total_duration = 0.0
            
            for i, segment_info in enumerate(raw_segments):
                if isinstance(segment_info, list) and len(segment_info) >= 2:
                    # Segments are in [start_ms, end_ms] format
                    start_ms, end_ms = segment_info[0], segment_info[1]
                    start_sec = start_ms / 1000.0
                    end_sec = end_ms / 1000.0
                    duration = end_sec - start_sec
                    
                    segments.append(VADSegment(
                        start=start_sec,
                        end=end_sec,
                        confidence=0.95,  # FunASR VAD is generally reliable
                    ))
                    
                    total_duration += duration

            logger.info(f"Detected {len(segments)} speech segments, total duration: {total_duration:.2f}s")

            return VADResult(
                segments=segments,
                total_speech_duration=total_duration,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            raise Exception(f"FunASR VAD processing failed: {str(e)}") from e
