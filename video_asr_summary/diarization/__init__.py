"""Speaker diarization processors."""

from .integrator import SegmentBasedIntegrator
from .pyannote_processor import PyannoteAudioProcessor

__all__ = ["PyannoteAudioProcessor", "SegmentBasedIntegrator"]
