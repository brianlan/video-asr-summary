"""Pipeline state management for saving and resuming processing."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, TYPE_CHECKING
from video_asr_summary.core import AudioData, TranscriptionResult

if TYPE_CHECKING:
    from video_asr_summary.analysis import AnalysisResult


@dataclass
class PipelineState:
    """Represents the current state of pipeline processing."""
    
    video_path: str
    output_dir: str
    started_at: str
    updated_at: str
    completed_steps: List[str]
    current_step: Optional[str]
    failed_step: Optional[str]
    error_message: Optional[str]
    
    # Intermediate file paths
    audio_file: str = ""
    transcription_file: str = ""
    analysis_file: str = ""
    final_result_file: str = ""
    
    # Processing parameters
    analysis_language: str = "en"
    content_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        """Create from dictionary loaded from JSON."""
        return cls(**data)


class StateManager:
    """Manages pipeline state and intermediate files."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize state manager with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / "pipeline_state.json"
        
        # Define intermediate file patterns
        self.file_patterns = {
            "audio": "audio.wav",
            "transcription": "transcription.json", 
            "analysis": "analysis.json",
            "final_result": "pipeline_result.json"
        }
    
    def create_state(
        self, 
        video_path: Union[str, Path],
        analysis_language: str = "en",
        content_type: Optional[str] = None
    ) -> PipelineState:
        """Create initial pipeline state."""
        now = datetime.now().isoformat()
        
        state = PipelineState(
            video_path=str(video_path),
            output_dir=str(self.output_dir),
            started_at=now,
            updated_at=now,
            completed_steps=[],
            current_step="initializing",
            failed_step=None,
            error_message=None,
            analysis_language=analysis_language,
            content_type=content_type
        )
        
        # Set intermediate file paths
        state.audio_file = str(self.output_dir / self.file_patterns["audio"])
        state.transcription_file = str(self.output_dir / self.file_patterns["transcription"])
        state.analysis_file = str(self.output_dir / self.file_patterns["analysis"])
        state.final_result_file = str(self.output_dir / self.file_patterns["final_result"])
        
        self.save_state(state)
        return state
    
    def load_state(self) -> Optional[PipelineState]:
        """Load existing pipeline state."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            return PipelineState.from_dict(data)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Could not load state file: {e}")
            return None
    
    def save_state(self, state: PipelineState) -> None:
        """Save pipeline state to disk."""
        state.updated_at = datetime.now().isoformat()
        
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
    
    def update_step(self, state: PipelineState, step_name: str) -> None:
        """Update current processing step."""
        state.current_step = step_name
        self.save_state(state)
    
    def complete_step(self, state: PipelineState, step_name: str) -> None:
        """Mark a step as completed."""
        if step_name not in state.completed_steps:
            state.completed_steps.append(step_name)
        state.current_step = None
        self.save_state(state)
    
    def fail_step(self, state: PipelineState, step_name: str, error: str) -> None:
        """Mark a step as failed."""
        state.failed_step = step_name
        state.error_message = error
        state.current_step = None
        self.save_state(state)
    
    def save_audio_data(self, state: PipelineState, audio_data: AudioData) -> None:
        """Save audio data metadata."""
        audio_metadata = {
            "file_path": str(audio_data.file_path),
            "duration_seconds": audio_data.duration_seconds,
            "sample_rate": audio_data.sample_rate,
            "channels": audio_data.channels,
            "format": audio_data.format,
            "saved_at": datetime.now().isoformat()
        }
        
        metadata_file = self.output_dir / "audio_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(audio_metadata, f, indent=2, ensure_ascii=False)
    
    def save_transcription(self, state: PipelineState, transcription: TranscriptionResult) -> None:
        """Save transcription result."""
        if not state.transcription_file:
            raise ValueError("Transcription file path not set in state")
            
        transcription_data = {
            "text": transcription.text,
            "confidence": transcription.confidence,
            "segments": transcription.segments,
            "language": transcription.language,
            "processing_time_seconds": transcription.processing_time_seconds,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(state.transcription_file, 'w', encoding='utf-8') as f:
            json.dump(transcription_data, f, indent=2, ensure_ascii=False)
    
    def save_analysis(self, state: PipelineState, analysis: "AnalysisResult") -> None:
        """Save analysis result."""
        if not state.analysis_file:
            raise ValueError("Analysis file path not set in state")
            
        # Convert AnalysisResult to serializable format
        analysis_data = {
            "content_type": analysis.content_type.value,
            "overall_credibility": analysis.overall_credibility,
            "response_language": analysis.response_language,
            "processing_time_seconds": analysis.processing_time_seconds,
            "timestamp": analysis.timestamp.isoformat() if analysis.timestamp else None,
            "conclusions": [
                {
                    "statement": c.statement,
                    "confidence": c.confidence,
                    "supporting_arguments": c.supporting_arguments,
                    "evidence_quality": c.evidence_quality,
                    "logical_issues": c.logical_issues
                }
                for c in analysis.conclusions
            ],
            "key_insights": analysis.key_insights,
            "potential_biases": analysis.potential_biases,
            "factual_claims": analysis.factual_claims,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(state.analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    def save_final_result(self, state: PipelineState, result: Dict[str, Any]) -> None:
        """Save final pipeline result."""
        if not state.final_result_file:
            raise ValueError("Final result file path not set in state")
            
        with open(state.final_result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def load_transcription(self, state: PipelineState) -> Optional[Dict[str, Any]]:
        """Load saved transcription data."""
        if not state.transcription_file or not os.path.exists(state.transcription_file):
            return None
        
        try:
            with open(state.transcription_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def load_analysis(self, state: PipelineState) -> Optional[Dict[str, Any]]:
        """Load saved analysis data."""
        if not state.analysis_file or not os.path.exists(state.analysis_file):
            return None
        
        try:
            with open(state.analysis_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    
    def is_step_completed(self, state: PipelineState, step_name: str) -> bool:
        """Check if a step has been completed."""
        return step_name in state.completed_steps
    
    def get_resume_point(self, state: PipelineState) -> str:
        """Determine where to resume processing."""
        if state.failed_step:
            return state.failed_step
        
        pipeline_steps = [
            "video_info_extraction",
            "audio_extraction", 
            "transcription",
            "analysis",
            "finalization"
        ]
        
        for step in pipeline_steps:
            if not self.is_step_completed(state, step):
                return step
        
        return "completed"
    
    def cleanup_intermediate_files(self, keep_final_result: bool = True) -> None:
        """Clean up intermediate files, optionally keeping final result."""
        files_to_remove = [
            self.output_dir / self.file_patterns["audio"],
            self.output_dir / "audio_metadata.json",
            self.state_file
        ]
        
        if not keep_final_result:
            files_to_remove.extend([
                self.output_dir / self.file_patterns["transcription"],
                self.output_dir / self.file_patterns["analysis"],
                self.output_dir / self.file_patterns["final_result"]
            ])
        
        for file_path in files_to_remove:
            if file_path.exists():
                try:
                    file_path.unlink()
                except OSError as e:
                    print(f"Warning: Could not remove {file_path}: {e}")
    
    def get_summary(self, state: PipelineState) -> Dict[str, Any]:
        """Get a summary of pipeline state."""
        return {
            "video_path": state.video_path,
            "output_dir": state.output_dir,
            "started_at": state.started_at,
            "updated_at": state.updated_at,
            "completed_steps": state.completed_steps,
            "current_step": state.current_step,
            "failed_step": state.failed_step,
            "error_message": state.error_message,
            "resume_point": self.get_resume_point(state),
            "analysis_language": state.analysis_language,
            "content_type": state.content_type
        }
