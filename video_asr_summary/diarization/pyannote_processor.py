"""pyannote.audio-based speaker diarization processor."""

import time
from pathlib import Path
from typing import Optional
import logging

from ..core import SpeakerDiarizationProcessor, DiarizationResult, SpeakerSegment


logger = logging.getLogger(__name__)


class PyannoteAudioProcessor(SpeakerDiarizationProcessor):
    """Speaker diarization using pyannote.audio."""
    
    def __init__(self, auth_token: str, device: str = "auto"):
        """
        Initialize the pyannote.audio processor.
        
        Args:
            auth_token: Hugging Face access token for model access
            device: Device to run on ("auto", "cpu", "cuda", "mps")
                   "auto" will choose the best available device
        """
        self.auth_token = auth_token
        self.device = self._select_device(device)
        self._pipeline = None
        
    def _select_device(self, device: str) -> str:
        """Select the best available device for processing."""
        if device == "auto":
            try:
                import torch
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    logger.info("Auto-selected MPS (Apple Silicon GPU) for diarization")
                    return "mps"
                elif torch.cuda.is_available():
                    logger.info("Auto-selected CUDA for diarization")
                    return "cuda"
                else:
                    logger.info("Auto-selected CPU for diarization")
                    return "cpu"
            except ImportError:
                logger.warning("PyTorch not available, defaulting to CPU")
                return "cpu"
        return device
    
    def _load_pipeline(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.auth_token
                )
                
                # Move to specified device if available
                if self.device in ["cuda", "mps"]:
                    try:
                        import torch
                        if self.device == "cuda" and torch.cuda.is_available():
                            self._pipeline.to(torch.device("cuda"))
                            logger.info("Using CUDA for diarization")
                        elif self.device == "mps" and torch.backends.mps.is_available():
                            self._pipeline.to(torch.device("mps"))
                            logger.info("Using MPS (Apple Silicon GPU) for diarization")
                        elif self.device == "cuda":
                            logger.warning("CUDA requested but not available, using CPU")
                        elif self.device == "mps":
                            logger.warning("MPS requested but not available, using CPU")
                    except ImportError:
                        logger.warning("PyTorch not available for GPU acceleration, using CPU")
                    except Exception as e:
                        logger.warning(f"Failed to move pipeline to {self.device}: {e}, using CPU")
                
                logger.info("Pyannote audio pipeline loaded successfully")
                
            except ImportError as e:
                raise ImportError(
                    "pyannote.audio is not installed. Install it with: pip install pyannote.audio"
                ) from e
            except Exception as e:
                raise RuntimeError(f"Failed to load pyannote.audio pipeline: {e}") from e
    
    def diarize(self, audio_path: Path, num_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Optional hint for number of speakers
            
        Returns:
            DiarizationResult with speaker segments
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If diarization fails
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        try:
            # Load pipeline on first use
            self._load_pipeline()
            
            # Apply diarization
            kwargs = {}
            if num_speakers is not None:
                kwargs["min_speakers"] = num_speakers
                kwargs["max_speakers"] = num_speakers
            
            logger.info(f"Starting diarization for {audio_path}")
            diarization = self._pipeline(str(audio_path), **kwargs)
            
            # Convert to our format
            segments = []
            speakers = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    confidence=1.0  # pyannote doesn't provide confidence scores
                )
                segments.append(segment)
                speakers.add(speaker)
            
            processing_time = time.time() - start_time
            
            result = DiarizationResult(
                segments=segments,
                num_speakers=len(speakers),
                processing_time_seconds=processing_time
            )
            
            logger.info(
                f"Diarization completed in {processing_time:.2f}s, "
                f"found {len(speakers)} speakers in {len(segments)} segments"
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {e}") from e
