"""
Simple demo of the complete video processing pipeline.

This demonstrates the pipeline system with state management and resumability.
"""

import os
from pathlib import Path
from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator


def main():
    """Demonstrate the pipeline functionality."""
    
    # Create a simple test setup
    output_dir = Path("./pipeline_demo_output")
    
    # Use a placeholder video path (in real usage this would be a real video file)
    video_path = Path("demo_video.mp4")
    
    print("🎬 VIDEO PROCESSING PIPELINE DEMO")
    print("=" * 40)
    print(f"📁 Output Directory: {output_dir}")
    print(f"📹 Video Path: {video_path} (placeholder)")
    print()
    
    # Check if LLM analysis is available
    has_llm = bool(os.getenv("OPENAI_ACCESS_TOKEN"))
    print(f"🧠 LLM Analysis: {'✅ Available' if has_llm else '❌ Not available (OPENAI_ACCESS_TOKEN not set)'}")
    print()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(output_dir)
    
    # Test state management
    print("📊 TESTING STATE MANAGEMENT")
    print("-" * 30)
    
    # Check if there's existing state
    existing_summary = orchestrator.get_state_summary()
    if existing_summary:
        print("📂 Found existing pipeline state:")
        print(f"   Video: {Path(existing_summary['video_path']).name}")
        print(f"   Started: {existing_summary['started_at']}")
        print(f"   Status: {existing_summary['resume_point']}")
        print(f"   Completed: {len(existing_summary['completed_steps'])} steps")
        print()
    else:
        print("📄 No existing pipeline state found")
        print()
    
    try:
        # Try to process video (this will use placeholder processors)
        print("🚀 STARTING PIPELINE PROCESSING")
        print("-" * 35)
        
        results = orchestrator.process_video(
            video_path=video_path,
            analysis_language="en",
            content_type=None,  # Auto-detect
            resume=True,
            cleanup_intermediate=False
        )
        
        print("\n📋 PROCESSING COMPLETED")
        print("=" * 25)
        
        # Show results summary
        if results.get("transcription"):
            print(f"📝 Transcription: ✅ ({len(results['transcription']['text'])} chars)")
        
        if results.get("analysis"):
            analysis = results["analysis"]
            print(f"🧠 Analysis: ✅ ({analysis['content_type']}, {analysis['overall_credibility']} credibility)")
        
        print(f"📁 Results saved to: {output_dir}")
        
        # List output files
        output_files = list(output_dir.glob("*.json"))
        if output_files:
            print("\n📄 Generated Files:")
            for file in output_files:
                print(f"   • {file.name}")
        
        print(f"\n💡 To view status: python -c \"from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator; print(PipelineOrchestrator('{output_dir}').get_state_summary())\"")
        
    except FileNotFoundError:
        print("⚠️  Video file not found (this is expected for the demo)")
        print("   In real usage, provide a valid video file path")
        
        # Still show state management capabilities
        print("\n📊 DEMONSTRATING STATE MANAGEMENT")
        print("-" * 35)
        
        # Create a demo state manually
        from video_asr_summary.pipeline.state_manager import StateManager
        
        state_manager = StateManager(output_dir)
        demo_state = state_manager.create_state(
            video_path="demo_video.mp4",
            analysis_language="en", 
            content_type="general"
        )
        
        print(f"✅ Created demo state: {demo_state.video_path}")
        print(f"   Output dir: {demo_state.output_dir}")
        print(f"   Started at: {demo_state.started_at}")
        print(f"   Analysis language: {demo_state.analysis_language}")
        
        # Update some steps
        state_manager.update_step(demo_state, "video_info_extraction")
        state_manager.complete_step(demo_state, "video_info_extraction")
        state_manager.update_step(demo_state, "audio_extraction")
        
        summary = state_manager.get_summary(demo_state)
        print("\n📈 State Summary:")
        print(f"   Completed steps: {summary['completed_steps']}")
        print(f"   Current step: {summary['current_step']}")
        print(f"   Resume point: {summary['resume_point']}")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        print("💾 State has been saved for debugging")
    
    print(f"\n🧹 To clean up demo files: rm -rf {output_dir}")


if __name__ == "__main__":
    main()
