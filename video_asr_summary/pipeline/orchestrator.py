"""Pipeline orchestrator for video processing workflow."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from video_asr_summary.core import VideoInfo, AudioData, TranscriptionResult
from video_asr_summary.pipeline.state_manager import StateManager, PipelineState

# Import actual processors
try:
    from video_asr_summary.video.opencv_processor import OpenCVVideoProcessor
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSOR_AVAILABLE = False

try:
    from video_asr_summary.audio.extractor import FFmpegAudioExtractor
    AUDIO_EXTRACTOR_AVAILABLE = True
except ImportError:
    AUDIO_EXTRACTOR_AVAILABLE = False

try:
    from video_asr_summary.asr.whisper_processor import WhisperProcessor
    from video_asr_summary.asr.funasr_processor import FunASRProcessor
    ASR_PROCESSOR_AVAILABLE = True
except ImportError:
    ASR_PROCESSOR_AVAILABLE = False

# Import for analysis
try:
    from video_asr_summary.analysis import AnalysisResult, ContentType
    from video_asr_summary.analysis.analyzer import DefaultContentAnalyzer
    from video_asr_summary.analysis.classifier import KeywordBasedClassifier
    from video_asr_summary.analysis.llm_client import OpenAICompatibleClient
    from video_asr_summary.analysis.prompt_templates import DefaultPromptTemplateManager
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


class PipelineOrchestrator:
    """Orchestrates the complete video processing pipeline with state management."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize pipeline orchestrator."""
        self.output_dir = Path(output_dir)
        self.state_manager = StateManager(self.output_dir)
        
        # Initialize real processors if available
        self._video_processor = None
        self._audio_extractor = None 
        self._asr_processor = None
        self._content_analyzer = None
        
        # Try to initialize video processor
        if VIDEO_PROCESSOR_AVAILABLE:
            try:
                self._video_processor = OpenCVVideoProcessor()
                print("✅ Video processor initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize video processor: {e}")
        
        # Try to initialize audio extractor
        if AUDIO_EXTRACTOR_AVAILABLE:
            try:
                self._audio_extractor = FFmpegAudioExtractor()
                print("✅ Audio extractor initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize audio extractor: {e}")
        
        # Try to initialize ASR processor  
        if ASR_PROCESSOR_AVAILABLE:
            try:
                # Initialize with default Whisper - will be updated based on language later
                self._asr_processor = WhisperProcessor()
                print("✅ ASR processor initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize ASR processor: {e}")
        
        # Initialize analysis components if available
        if ANALYSIS_AVAILABLE and os.getenv("OPENAI_ACCESS_TOKEN"):
            try:
                llm_client = OpenAICompatibleClient()
                template_manager = DefaultPromptTemplateManager()
                classifier = KeywordBasedClassifier()
                self._content_analyzer = DefaultContentAnalyzer(
                    llm_client, template_manager, classifier
                )
                print("✅ Content analyzer initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize content analyzer: {e}")
                self._content_analyzer = None
    
    def set_processors(
        self,
        video_processor=None,
        audio_extractor=None,
        asr_processor=None
    ):
        """Set the actual processor instances."""
        if video_processor:
            self._video_processor = video_processor
        if audio_extractor:
            self._audio_extractor = audio_extractor
        if asr_processor:
            self._asr_processor = asr_processor
    
    def process_video(
        self,
        video_path: Union[str, Path],
        analysis_language: str = "en",
        content_type: Optional[str] = None,
        resume: bool = True,
        cleanup_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Process video through complete pipeline with state management.
        
        Args:
            video_path: Path to input video file
            analysis_language: Language for content analysis response
            content_type: Optional content type override for analysis
            resume: Whether to resume from existing state
            cleanup_intermediate: Whether to clean up intermediate files when done
            
        Returns:
            Dictionary with complete processing results
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Load or create state
        state = None
        if resume:
            state = self.state_manager.load_state()
        
        if state is None or state.video_path != str(video_path):
            print(f"🎬 Starting new pipeline for: {video_path.name}")
            state = self.state_manager.create_state(
                video_path, analysis_language, content_type
            )
        else:
            print(f"📂 Resuming pipeline from: {self.state_manager.get_resume_point(state)}")
        
        try:
            # Execute pipeline steps
            video_info = self._extract_video_info(state, video_path)
            audio_data = self._extract_audio(state, video_path)
            transcription = self._transcribe_audio(state, audio_data)
            analysis = self._analyze_content(state, transcription)
            
            # Create final result
            result = self._finalize_results(state, {
                "video_info": self._video_info_to_dict(video_info) if video_info else None,
                "audio_data": self._audio_data_to_dict(audio_data) if audio_data else None,
                "transcription": self._transcription_to_dict(transcription) if transcription else None,
                "analysis": self._analysis_to_dict(analysis) if analysis else None,
                "processing_summary": self.state_manager.get_summary(state)
            })
            
            print("✅ Pipeline completed successfully!")
            
            # Cleanup if requested
            if cleanup_intermediate:
                print("🧹 Cleaning up intermediate files...")
                self.state_manager.cleanup_intermediate_files(keep_final_result=True)
            
            return result
        
        except Exception as e:
            self.state_manager.fail_step(state, state.current_step or "unknown", str(e))
            print(f"❌ Pipeline failed at step '{state.current_step}': {e}")
            raise
    
    def _extract_video_info(self, state: PipelineState, video_path: Path) -> Optional[VideoInfo]:
        """Extract video information."""
        step_name = "video_info_extraction"
        
        if self.state_manager.is_step_completed(state, step_name):
            print("⏭️  Video info extraction already completed")
            return None
        
        print("📹 Extracting video information...")
        self.state_manager.update_step(state, step_name)
        
        if not self._video_processor:
            print("⚠️  Video processor not available, using basic file info")
            # Basic file info as fallback
            try:
                file_stats = video_path.stat()
                video_info = VideoInfo(
                    file_path=video_path,
                    duration_seconds=0.0,  # Unknown without processing
                    frame_rate=0.0,       # Unknown without processing
                    width=0,              # Unknown without processing
                    height=0,             # Unknown without processing
                    file_size_bytes=file_stats.st_size
                )
                self.state_manager.complete_step(state, step_name)
                print(f"✅ Basic file info extracted: {video_info.file_size_bytes} bytes")
                return video_info
            except Exception as e:
                self.state_manager.fail_step(state, step_name, str(e))
                raise
        
        try:
            # Use real video processor
            video_info = self._video_processor.extract_info(video_path)
            
            self.state_manager.complete_step(state, step_name)
            print(f"✅ Video info extracted: {video_info.duration_seconds:.1f}s, {video_info.width}x{video_info.height}")
            return video_info
            
        except Exception as e:
            self.state_manager.fail_step(state, step_name, str(e))
            raise
    
    def _extract_audio(self, state: PipelineState, video_path: Path) -> Optional[AudioData]:
        """Extract audio from video."""
        step_name = "audio_extraction"
        
        if self.state_manager.is_step_completed(state, step_name):
            print("⏭️  Audio extraction already completed")
            # Load audio metadata if available
            return self._load_audio_metadata(state)
        
        print("🎵 Extracting audio from video...")
        self.state_manager.update_step(state, step_name)
        
        if not self._audio_extractor:
            print("⚠️  Audio extractor not available, skipping audio extraction")
            self.state_manager.complete_step(state, step_name)
            return None
        
        try:
            # Extract audio using real extractor
            audio_path = Path(state.audio_file)
            audio_data = self._audio_extractor.extract_audio(
                video_path=video_path,
                output_path=audio_path,
                sample_rate=16000,  # Good for ASR
                channels=1,         # Mono for ASR
                format="wav"
            )
            
            # Save audio metadata
            self.state_manager.save_audio_data(state, audio_data)
            self.state_manager.complete_step(state, step_name)
            print(f"✅ Audio extracted: {audio_data.duration_seconds:.1f}s, {audio_data.sample_rate}Hz")
            return audio_data
            
        except Exception as e:
            self.state_manager.fail_step(state, step_name, str(e))
            raise
    
    def _transcribe_audio(self, state: PipelineState, audio_data: Optional[AudioData]) -> Optional[TranscriptionResult]:
        """Transcribe audio to text."""
        step_name = "transcription"
        
        if self.state_manager.is_step_completed(state, step_name):
            print("⏭️  Transcription already completed")
            return self._load_transcription_result(state)
        
        if not audio_data:
            print("⚠️  No audio data available, skipping transcription")
            self.state_manager.complete_step(state, step_name)
            return None
        
        print("🎙️  Transcribing audio to text...")
        self.state_manager.update_step(state, step_name)
        
        # Get appropriate ASR processor based on analysis language
        analysis_language = getattr(state, 'analysis_language', 'en')
        asr_processor = self._get_asr_processor(analysis_language)
        
        if not asr_processor:
            print("⚠️  ASR processor not available, using placeholder transcription")
            # Create placeholder transcription for demo
            transcription = TranscriptionResult(
                text="This is a placeholder transcription for demonstration purposes.",
                confidence=0.85,
                segments=[
                    {"start": 0.0, "end": 5.0, "text": "This is a placeholder transcription", "confidence": 0.9},
                    {"start": 5.0, "end": 10.0, "text": "for demonstration purposes.", "confidence": 0.8}
                ],
                language="en",
                processing_time_seconds=2.5
            )
        else:
            try:
                # Use language-appropriate ASR processor
                transcription = asr_processor.transcribe(audio_data.file_path)
            except Exception as e:
                self.state_manager.fail_step(state, step_name, str(e))
                raise
        
        try:
            self.state_manager.save_transcription(state, transcription)
            self.state_manager.complete_step(state, step_name)
            print(f"✅ Transcription completed: {len(transcription.text)} characters, {transcription.confidence:.2f} confidence")
            return transcription
            
        except Exception as e:
            self.state_manager.fail_step(state, step_name, str(e))
            raise
    
    def _analyze_content(self, state: PipelineState, transcription: Optional[TranscriptionResult]) -> Optional[Any]:
        """Analyze transcribed content."""
        step_name = "analysis"
        
        if self.state_manager.is_step_completed(state, step_name):
            print("⏭️  Content analysis already completed")
            return self._load_analysis_result(state)
        
        if not transcription or not transcription.text.strip():
            print("⚠️  No transcription available, skipping content analysis")
            self.state_manager.complete_step(state, step_name)
            return None
        
        if not self._content_analyzer:
            print("⚠️  Content analyzer not available, skipping analysis")
            self.state_manager.complete_step(state, step_name)
            return None
        
        print("🧠 Analyzing content with LLM...")
        self.state_manager.update_step(state, step_name)
        
        try:
            # Determine content type
            content_type = None
            if state.content_type and ANALYSIS_AVAILABLE:
                try:
                    content_type = ContentType(state.content_type)
                except ValueError:
                    print(f"⚠️  Invalid content type '{state.content_type}', using auto-detection")
            
            # Perform analysis
            start_time = time.time()
            analysis = self._content_analyzer.analyze(
                transcription.text,
                content_type=content_type,
                response_language=state.analysis_language
            )
            
            self.state_manager.save_analysis(state, analysis)
            self.state_manager.complete_step(state, step_name)
            
            elapsed = time.time() - start_time
            print(f"✅ Content analysis completed in {elapsed:.1f}s")
            print(f"   📊 Content type: {analysis.content_type.value}")
            print(f"   🎯 Overall credibility: {analysis.overall_credibility}")
            print(f"   🔍 Found {len(analysis.conclusions)} conclusions")
            
            return analysis
            
        except Exception as e:
            self.state_manager.fail_step(state, step_name, str(e))
            raise
    
    def _finalize_results(self, state: PipelineState, results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize and save complete results."""
        step_name = "finalization"
        
        print("📋 Finalizing results...")
        self.state_manager.update_step(state, step_name)
        
        try:
            # Add metadata
            final_results = {
                **results,
                "pipeline_info": {
                    "completed_at": state.updated_at,
                    "total_processing_time": state.updated_at,  # Could calculate actual time
                    "pipeline_version": "1.0.0",
                    "analysis_language": state.analysis_language,
                    "content_type": state.content_type
                }
            }
            
            self.state_manager.save_final_result(state, final_results)
            self.state_manager.complete_step(state, step_name)
            
            return final_results
            
        except Exception as e:
            self.state_manager.fail_step(state, step_name, str(e))
            raise
    
    # Helper methods for loading saved data
    def _load_audio_metadata(self, state: PipelineState) -> Optional[AudioData]:
        """Load audio metadata from saved file."""
        metadata_file = self.output_dir / "audio_metadata.json"
        if not metadata_file.exists():
            return None
        
        try:
            import json
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            return AudioData(
                file_path=Path(data["file_path"]),
                duration_seconds=data["duration_seconds"],
                sample_rate=data["sample_rate"],
                channels=data["channels"],
                format=data["format"]
            )
        except (json.JSONDecodeError, KeyError):
            return None
    
    def _load_transcription_result(self, state: PipelineState) -> Optional[TranscriptionResult]:
        """Load transcription result from saved file."""
        data = self.state_manager.load_transcription(state)
        if not data:
            return None
        
        return TranscriptionResult(
            text=data["text"],
            confidence=data["confidence"],
            segments=data["segments"],
            language=data.get("language"),
            processing_time_seconds=data.get("processing_time_seconds")
        )
    
    def _load_analysis_result(self, state: PipelineState) -> Optional[Dict[str, Any]]:
        """Load analysis result from saved file."""
        return self.state_manager.load_analysis(state)
    
    # Helper methods for converting to dictionaries
    def _video_info_to_dict(self, video_info: VideoInfo) -> Dict[str, Any]:
        """Convert VideoInfo to dictionary."""
        return {
            "file_path": str(video_info.file_path),
            "duration_seconds": video_info.duration_seconds,
            "frame_rate": video_info.frame_rate,
            "width": video_info.width,
            "height": video_info.height,
            "file_size_bytes": video_info.file_size_bytes
        }
    
    def _audio_data_to_dict(self, audio_data: AudioData) -> Dict[str, Any]:
        """Convert AudioData to dictionary."""
        return {
            "file_path": str(audio_data.file_path),
            "duration_seconds": audio_data.duration_seconds,
            "sample_rate": audio_data.sample_rate,
            "channels": audio_data.channels,
            "format": audio_data.format
        }
    
    def _transcription_to_dict(self, transcription: TranscriptionResult) -> Dict[str, Any]:
        """Convert TranscriptionResult to dictionary."""
        return {
            "text": transcription.text,
            "confidence": transcription.confidence,
            "segments": transcription.segments,
            "language": transcription.language,
            "processing_time_seconds": transcription.processing_time_seconds
        }
    
    def _analysis_to_dict(self, analysis: Any) -> Optional[Dict[str, Any]]:
        """Convert AnalysisResult to dictionary."""
        if not analysis:
            return None
        
        if hasattr(analysis, 'content_type'):
            # It's an AnalysisResult object
            return {
                "content_type": analysis.content_type.value,
                "overall_credibility": analysis.overall_credibility,
                "response_language": analysis.response_language,
                "processing_time_seconds": analysis.processing_time_seconds,
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
                "factual_claims": analysis.factual_claims
            }
        else:
            # It's already a dict (loaded from file)
            return analysis
    
    def get_state_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current pipeline state."""
        state = self.state_manager.load_state()
        if not state:
            return None
        
        return self.state_manager.get_summary(state)
    
    def cleanup(self, keep_final_result: bool = True) -> None:
        """Clean up intermediate files."""
        self.state_manager.cleanup_intermediate_files(keep_final_result)
    
    def _get_asr_processor(self, language: str):
        """Get appropriate ASR processor based on language.
        
        Args:
            language: Target language ('zh', 'en', etc.)
            
        Returns:
            ASR processor instance appropriate for the language
        """
        if not ASR_PROCESSOR_AVAILABLE:
            return None
            
        # Use FunASR for Chinese languages, Whisper for others
        if language.lower() in ['zh', 'zh-cn', 'zh-tw', 'chinese', 'mandarin']:
            try:
                print("🇨🇳 Using FunASR processor for Chinese language")
                # Use Chinese-specific FunASR model with better punctuation
                return FunASRProcessor(
                    model_path="iic/SenseVoiceSmall",
                    language="zn",  # FunASR Chinese language code
                    device="cpu"    # Use CPU for better compatibility
                )
            except Exception as e:
                print(f"⚠️  Could not initialize FunASR processor: {e}")
                print("🔄 Falling back to Whisper processor")
                return WhisperProcessor(language="zh")
        else:
            print("🌍 Using Whisper processor for non-Chinese language")
            # Map common language codes for Whisper
            whisper_lang = language.lower()
            if whisper_lang in ['en', 'english']:
                whisper_lang = "en"
            elif whisper_lang in ['ja', 'japanese']:
                whisper_lang = "ja"
            elif whisper_lang in ['ko', 'korean']:
                whisper_lang = "ko"
            else:
                whisper_lang = None  # Let Whisper auto-detect
                
            return WhisperProcessor(language=whisper_lang)
