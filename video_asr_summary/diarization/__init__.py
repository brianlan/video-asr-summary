"""Speaker diarization processors."""

from .pyannote_processor import PyannoteAudioProcessor
from .integrator import SegmentBasedIntegrator

__all__ = ["PyannoteAudioProcessor", "SegmentBasedIntegrator"]
