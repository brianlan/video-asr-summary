"""Types and interfaces for Voice Activity Detection (VAD)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class VADSegment:
    """A segment detected by Voice Activity Detection."""
    
    start: float  # Start time in seconds
    end: float    # End time in seconds
    confidence: float = 1.0  # Confidence score for voice activity


@dataclass
class VADResult:
    """Result of Voice Activity Detection."""
    
    segments: List[VADSegment]
    total_speech_duration: float  # Total speech time in seconds
    processing_time_seconds: Optional[float] = None


class VADProcessor(ABC):
    """Abstract base class for Voice Activity Detection."""
    
    @abstractmethod
    def detect_voice_activity(self, audio_path: Path) -> VADResult:
        """Detect voice activity in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            VADResult with detected speech segments
        """
        pass
