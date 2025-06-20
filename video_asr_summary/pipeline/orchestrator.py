"""Pipeline orchestrator for video processing workflow."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from video_asr_summary.core import VideoInfo, AudioData, TranscriptionResult, DiarizationResult, EnhancedTranscriptionResult
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
    from video_asr_summary.integration.specialized_asr_integrator import SpecializedASRIntegrator
    ASR_PROCESSOR_AVAILABLE = True
except ImportError:
    ASR_PROCESSOR_AVAILABLE = False

# Import for diarization
try:
    from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor
    from video_asr_summary.diarization.integrator import SegmentBasedIntegrator
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

# Import for analysis
try:
    from video_asr_summary.analysis import ContentType
    from video_asr_summary.analysis.analyzer import DefaultContentAnalyzer
    from video_asr_summary.analysis.classifier import KeywordBasedClassifier
    from video_asr_summary.analysis.llm_client import OpenAICompatibleClient
    from video_asr_summary.analysis.prompt_templates import DefaultPromptTemplateManager
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False


class PipelineOrchestrator:
    """Orchestrates the complete video processing pipeline with state management."""
    
    def __init__(
        self, 
        output_dir: Union[str, Path],
        llm_model: str = "deepseek-chat",
        llm_endpoint: str = "https://api.deepseek.com/v1", 
        llm_timeout: int = 1200
    ):
        """Initialize pipeline orchestrator."""
        self.output_dir = Path(output_dir)
        self.state_manager = StateManager(self.output_dir)
        
        # Store LLM configuration
        self.llm_model = llm_model
        self.llm_endpoint = llm_endpoint
        self.llm_timeout = llm_timeout
        
        # Initialize real processors if available
        self._video_processor = None
        self._audio_extractor = None 
        self._asr_processor = None
        self._diarization_processor = None
        self._diarization_integrator = None
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
        
        # Try to initialize diarization components
        if DIARIZATION_AVAILABLE:
            try:
                # Check for Hugging Face token for pyannote
                hf_token = os.getenv("HF_ACCESS_TOKEN")
                if hf_token:
                    self._diarization_processor = PyannoteAudioProcessor(
                        auth_token=hf_token,
                        device="auto"  # Auto-select best device (MPS/CUDA/CPU)
                    )
                    self._diarization_integrator = SegmentBasedIntegrator()
                    print("✅ Diarization processor initialized")
                    print("   🎙️ Using pyannote.audio for speaker diarization")
                else:
                    print("⚠️  Diarization disabled: HF_ACCESS_TOKEN not found")
                    print("   💡 Set HF_ACCESS_TOKEN environment variable for speaker diarization")
            except Exception as e:
                print(f"⚠️  Could not initialize diarization: {e}")
                self._diarization_processor = None
                self._diarization_integrator = None
        
        # Initialize analysis components if available
        if ANALYSIS_AVAILABLE and os.getenv("OPENAI_ACCESS_TOKEN"):
            try:
                llm_client = OpenAICompatibleClient(
                    model=self.llm_model,
                    base_url=self.llm_endpoint,
                    timeout=self.llm_timeout
                )
                template_manager = DefaultPromptTemplateManager()
                classifier = KeywordBasedClassifier()
                self._content_analyzer = DefaultContentAnalyzer(
                    llm_client, template_manager, classifier
                )
                print("✅ Content analyzer initialized")
                print(f"   📡 Model: {self.llm_model}")
                print(f"   🌐 Endpoint: {self.llm_endpoint}")
                print(f"   ⏱️ Timeout: {self.llm_timeout}s")
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
            
            # Optimization: Skip separate diarization if using SpecializedASRIntegrator
            # since it does VAD + ASR + Punctuation + Diarization internally
            if self._is_using_specialized_asr(analysis_language):
                print("🔧 Using SpecializedASRIntegrator (Chinese-optimized) - skipping separate diarization step")
                diarization = None
                transcription = self._transcribe_audio(state, audio_data)
                # Check if enhanced result is available from SpecializedASRIntegrator
                enhanced_transcription = self.state_manager.load_enhanced_transcription(state)
                if enhanced_transcription is None:
                    # Fallback: create enhanced transcription without speaker info
                    enhanced_transcription = self._integrate_diarization(state, transcription, None)
            else:
                print("🔧 Using Whisper + Pyannote pipeline with separate diarization")
                diarization = self._diarize_speakers(state, audio_data)
                transcription = self._transcribe_audio(state, audio_data)
                enhanced_transcription = self._integrate_diarization(state, transcription, diarization)
            
            analysis = self._analyze_content(state, enhanced_transcription)
            
            # Create final result
            result = self._finalize_results(state, {
                "video_info": self._video_info_to_dict(video_info) if video_info else None,
                "audio_data": self._audio_data_to_dict(audio_data) if audio_data else None,
                "diarization": self._diarization_to_dict(diarization) if diarization else None,
                "transcription": self._transcription_to_dict(transcription) if transcription else None,
                "enhanced_transcription": self._enhanced_transcription_to_dict(enhanced_transcription) if enhanced_transcription else None,
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
            enhanced_result = None
            try:
                # Check if it's the specialized integrator
                if hasattr(asr_processor, 'process_audio'):
                    # Use the specialized 4-model pipeline
                    enhanced_result = asr_processor.process_audio(audio_data.file_path)  # type: ignore
                    # Extract the transcription part for the traditional pipeline
                    transcription = enhanced_result.transcription
                    print("✅ Used SpecializedASRIntegrator (4-model pipeline)")
                else:
                    # Use traditional ASR processor
                    transcription = asr_processor.transcribe(audio_data.file_path)  # type: ignore
            except Exception as e:
                self.state_manager.fail_step(state, step_name, str(e))
                raise
        
        try:
            # Save transcription (for backward compatibility)
            self.state_manager.save_transcription(state, transcription)
            
            # Save enhanced result if available
            if enhanced_result is not None:
                self.state_manager.save_enhanced_transcription(state, enhanced_result)
                print("✅ Enhanced transcription result saved")
            
            self.state_manager.complete_step(state, step_name)
            print(f"✅ Transcription completed: {len(transcription.text)} characters, {transcription.confidence:.2f} confidence")
            return transcription
            
        except Exception as e:
            self.state_manager.fail_step(state, step_name, str(e))
            raise
    
    def _analyze_content(self, state: PipelineState, transcription: Optional[Union[TranscriptionResult, EnhancedTranscriptionResult]]) -> Optional[Any]:
        """Analyze transcribed content."""
        step_name = "analysis"
        
        if self.state_manager.is_step_completed(state, step_name):
            print("⏭️  Content analysis already completed")
            return self._load_analysis_result(state)
        
        if not transcription:
            print("⚠️  No transcription available, skipping content analysis")
            self.state_manager.complete_step(state, step_name)
            return None
        
        # Extract text from transcription (handle both types)
        if isinstance(transcription, EnhancedTranscriptionResult):
            # Format text with speaker attribution for better analysis
            speaker_segments = []
            for seg in transcription.speaker_attributed_segments:
                speaker = seg.get('speaker', 'Unknown Speaker')
                text_content = seg.get('text', '').strip()
                if text_content:
                    speaker_segments.append(f"{speaker}: {text_content}")
            
            if speaker_segments:
                text = "\n".join(speaker_segments)
                print(f"   🎙️ Using speaker-attributed text with {len(speaker_segments)} segments")
            else:
                # Fallback to regular text if no speaker segments
                text = transcription.transcription.text
                print("   ⚠️ No speaker segments found, using plain transcription")
        else:
            text = transcription.text
            
        if not text.strip():
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
                text,
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
    
    def _diarize_speakers(self, state: PipelineState, audio_data: Optional[AudioData]) -> Optional[DiarizationResult]:
        """Perform speaker diarization on audio."""
        step_name = "diarization"
        
        if self.state_manager.is_step_completed(state, step_name):
            print("⏭️  Speaker diarization already completed")
            return self._load_diarization_result(state)
        
        if not audio_data:
            print("⚠️  No audio data available, skipping diarization")
            self.state_manager.complete_step(state, step_name)
            return None
        
        if not self._diarization_processor:
            print("⚠️  Diarization processor not available, skipping speaker diarization")
            self.state_manager.complete_step(state, step_name)
            return None
        
        print("🎙️ Performing speaker diarization...")
        self.state_manager.update_step(state, step_name)
        
        try:
            start_time = time.time()
            diarization = self._diarization_processor.diarize(audio_data.file_path)
            processing_time = time.time() - start_time
            
            # Update processing time
            diarization.processing_time_seconds = processing_time
            
            # TODO: Add state manager support for diarization
            # self.state_manager.save_diarization(state, diarization)
            self.state_manager.complete_step(state, step_name)
            
            print(f"✅ Speaker diarization completed in {processing_time:.1f}s")
            print(f"   👥 Found {diarization.num_speakers} speakers")
            print(f"   📊 {len(diarization.segments)} speaker segments")
            
            return diarization
            
        except Exception as e:
            self.state_manager.fail_step(state, step_name, str(e))
            raise

    def _integrate_diarization(self, state: PipelineState, transcription: Optional[TranscriptionResult], 
                              diarization: Optional[DiarizationResult]) -> Optional[EnhancedTranscriptionResult]:
        """Integrate diarization results with transcription."""
        step_name = "diarization_integration"
        
        if self.state_manager.is_step_completed(state, step_name):
            print("⏭️  Diarization integration already completed")
            return self._load_enhanced_transcription_result(state)
        
        if not transcription:
            print("⚠️  No transcription available, skipping diarization integration")
            self.state_manager.complete_step(state, step_name)
            return None
        
        if not diarization:
            print("⚠️  No diarization results, creating enhanced transcription without speaker info")
            # Create enhanced transcription without speaker information
            enhanced = EnhancedTranscriptionResult(
                transcription=transcription,
                diarization=DiarizationResult(segments=[], num_speakers=0),
                speaker_attributed_segments=[
                    {
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': seg.get('text', ''),
                        'speaker': None,
                        'confidence': seg.get('confidence', 1.0)
                    }
                    for seg in transcription.segments
                ],
                processing_time_seconds=0.0
            )
            # TODO: Add state manager support for enhanced transcription  
            # self.state_manager.save_enhanced_transcription(state, enhanced)
            self.state_manager.complete_step(state, step_name)
            return enhanced
        
        if not self._diarization_integrator:
            print("⚠️  Diarization integrator not available")
            self.state_manager.complete_step(state, step_name)
            return None
        
        print("🔗 Integrating speaker diarization with transcription...")
        self.state_manager.update_step(state, step_name)
        
        try:
            start_time = time.time()
            enhanced_transcription = self._diarization_integrator.integrate(transcription, diarization)
            processing_time = time.time() - start_time
            
            # Update processing time
            enhanced_transcription.processing_time_seconds = processing_time
            
            # TODO: Add state manager support for enhanced transcription
            # self.state_manager.save_enhanced_transcription(state, enhanced_transcription)
            self.state_manager.complete_step(state, step_name)
            
            print(f"✅ Diarization integration completed in {processing_time:.1f}s")
            print(f"   📝 {len(enhanced_transcription.speaker_attributed_segments)} segments with speaker attribution")
            
            return enhanced_transcription
            
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
    
    def _is_using_specialized_asr(self, language: str) -> bool:
        """Check if we'll be using SpecializedASRIntegrator for the given language."""
        if not ASR_PROCESSOR_AVAILABLE:
            return False
        
        # Check what processor would be returned
        asr_processor = self._get_asr_processor(language)
        return hasattr(asr_processor, 'process_audio')

    def _get_asr_processor(self, language: str) -> Optional[Union['WhisperProcessor', 'SpecializedASRIntegrator']]:
        """Get appropriate ASR processor based on language.
        
        Args:
            language: Target language ('zh', 'en', etc.)
            
        Returns:
            ASR processor instance appropriate for the language
        """
        if not ASR_PROCESSOR_AVAILABLE:
            return None
        
        # Language-aware ASR processor selection
        language = language.lower()
        
        # Use SpecializedASRIntegrator (FunASR-based) for Chinese and related languages
        if language in ['zh', 'zh-cn', 'zh-tw', 'chinese', 'mandarin']:
            try:
                print("🔧 Using SpecializedASRIntegrator (4-model pipeline) for Chinese")
                # The specialized integrator handles VAD, ASR, punctuation, and diarization
                return SpecializedASRIntegrator(device="auto")
            except Exception as e:
                print(f"⚠️  Could not initialize SpecializedASRIntegrator: {e}")
                print("🔄 Falling back to Whisper processor")
        
        # Use Whisper for English and other languages (more robust for long audio)
        print(f"🔧 Using Whisper processor for language: {language}")
        try:
            # Map common language codes for Whisper
            whisper_lang = language.lower()
            if whisper_lang in ['en', 'english']:
                whisper_lang = "en"
            elif whisper_lang in ['ja', 'japanese']:
                whisper_lang = "ja"
            elif whisper_lang in ['ko', 'korean']:
                whisper_lang = "ko"
            elif whisper_lang in ['zh', 'zh-cn', 'zh-tw', 'chinese', 'mandarin']:
                whisper_lang = "zh"
            else:
                whisper_lang = None  # Let Whisper auto-detect
                
            return WhisperProcessor(language=whisper_lang)
        except Exception as e:
            print(f"⚠️  Could not initialize Whisper processor: {e}")
            return None
    
    def _diarization_to_dict(self, diarization: DiarizationResult) -> Dict[str, Any]:
        """Convert DiarizationResult to dictionary."""
        return {
            "num_speakers": diarization.num_speakers,
            "processing_time_seconds": diarization.processing_time_seconds,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "speaker": seg.speaker,
                    "confidence": seg.confidence
                }
                for seg in diarization.segments
            ]
        }
    
    def _enhanced_transcription_to_dict(self, enhanced: EnhancedTranscriptionResult) -> Dict[str, Any]:
        """Convert EnhancedTranscriptionResult to dictionary."""
        return {
            "transcription": self._transcription_to_dict(enhanced.transcription),
            "diarization": self._diarization_to_dict(enhanced.diarization),
            "speaker_attributed_segments": enhanced.speaker_attributed_segments,
            "processing_time_seconds": enhanced.processing_time_seconds
        }
    
    def _load_diarization_result(self, state: PipelineState) -> Optional[DiarizationResult]:
        """Load diarization result from saved file."""
        # TODO: Implement when state manager supports diarization
        return None
    
    def _load_enhanced_transcription_result(self, state: PipelineState) -> Optional[EnhancedTranscriptionResult]:
        """Load enhanced transcription result from saved file."""
        return self.state_manager.load_enhanced_transcription(state)
