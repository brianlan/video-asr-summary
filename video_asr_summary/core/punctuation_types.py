"""Types and interfaces for Punctuation Restoration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class PunctuationResult:
    """Result of punctuation restoration."""
    
    text: str  # Text with restored punctuation
    confidence: float  # Overall confidence score
    processing_time_seconds: Optional[float] = None


class PunctuationProcessor(ABC):
    """Abstract base class for Punctuation Restoration."""
    
    @abstractmethod
    def restore_punctuation(self, text: str) -> PunctuationResult:
        """Restore punctuation in text.
        
        Args:
            text: Input text without proper punctuation
            
        Returns:
            PunctuationResult with restored punctuation
        """
        pass
