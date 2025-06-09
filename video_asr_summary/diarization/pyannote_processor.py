"""pyannote.audio-based speaker diarization processor."""

import time
from pathlib import Path
from typing import Optional
import logging

from ..core import SpeakerDiarizationProcessor, DiarizationResult, SpeakerSegment


logger = logging.getLogger(__name__)


class PyannoteAudioProcessor(SpeakerDiarizationProcessor):
    """Speaker diarization using pyannote.audio."""
    
    def __init__(self, auth_token: str, device: str = "cpu"):
        """
        Initialize the pyannote.audio processor.
        
        Args:
            auth_token: Hugging Face access token for model access
            device: Device to run on ("cpu" or "cuda")
        """
        self.auth_token = auth_token
        self.device = device
        self._pipeline = None
    
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
                if self.device == "cuda":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            self._pipeline.to(torch.device("cuda"))
                            logger.info("Using CUDA for diarization")
                        else:
                            logger.warning("CUDA requested but not available, using CPU")
                    except ImportError:
                        logger.warning("PyTorch not available for CUDA, using CPU")
                
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
