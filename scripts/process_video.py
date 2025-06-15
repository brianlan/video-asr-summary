#!/usr/bin/env python3
"""
Complete video processing pipeline script.

This script processes video files through the complete pipeline:
1. Extract video information
2. Extract audio from video  
3. Transcribe audio to text using ASR
4. Analyze content with LLM for conclusions and arguments
5. Generate comprehensive results

Features:
- Resume from interruptions with state management
- Save intermediate files for debugging and iterative development
- Multi-language analysis support
- Content type auto-detection or manual specification
- Cleanup options for intermediate files
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process video files through complete ASR and analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/process_video.py input.mp4 ./output

  # With Spanish analysis and political content type  
  python scripts/process_video.py interview.mp4 ./results \\
    --analysis-language es --content-type political_commentary

  # With custom LLM model and endpoint
  python scripts/process_video.py video.mp4 ./output \\
    --llm-model "gpt-4" --llm-endpoint "https://api.openai.com/v1"

  # With custom Chinese model and timeout
  python scripts/process_video.py chinese_video.mp4 ./output \\
    --analysis-language zh --llm-model "qwen-max" \\
    --llm-endpoint "https://dashscope.aliyuncs.com/compatible-mode/v1" \\
    --llm-timeout 300

  # Resume interrupted processing
  python scripts/process_video.py video.mp4 ./output --resume

  # Clean up intermediate files when done
  python scripts/process_video.py video.mp4 ./output --cleanup

Content Types:
  - political_commentary: Political discussions and policy analysis
  - news_report: News articles and journalistic content
  - technical_review: Technical analysis and product reviews
  - book_section: Academic content and educational material
  - personal_casual_talk: Informal conversations and personal stories
  - general: General content (auto-detected if not specified)

Analysis Languages:
  en (English), es (Spanish), fr (French), de (German), it (Italian),
  pt (Portuguese), ru (Russian), ja (Japanese), ko (Korean), 
  zh (Chinese), ar (Arabic), hi (Hindi)

Popular LLM Models:
  - gemini-2.5-pro-preview-03-25 (default, good for Chinese)
  - gpt-4, gpt-4-turbo, gpt-3.5-turbo (OpenAI)
  - qwen-max, qwen-plus (Alibaba Cloud)
  - glm-4, glm-3-turbo (Zhipu AI)
  - deepseek-chat (DeepSeek)

Common Endpoints:
  - https://openai.newbotai.cn/v1 (default, multi-model)
  - https://api.openai.com/v1 (OpenAI official)
  - https://dashscope.aliyuncs.com/compatible-mode/v1 (Alibaba)
  - https://api.deepseek.com/v1 (DeepSeek)
        """
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "output_dir", 
        type=str,
        help="Output directory for results and intermediate files"
    )
    
    parser.add_argument(
        "--analysis-language", "-l",
        type=str,
        default="en",
        choices=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"],
        help="Language for analysis response (default: en)"
    )
    
    parser.add_argument(
        "--content-type", "-t",
        type=str,
        choices=[
            "political_commentary", "news_report", "technical_review",
            "book_section", "personal_casual_talk", "general"
        ],
        help="Content type for analysis (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--llm-model", "-m",
        type=str,
        default="gemini-2.5-pro-preview-03-25",
        help="LLM model name for content analysis (default: gemini-2.5-pro-preview-03-25)"
    )
    
    parser.add_argument(
        "--llm-endpoint", "-e",
        type=str,
        default="https://openai.newbotai.cn/v1",
        help="LLM API endpoint URL (default: https://openai.newbotai.cn/v1)"
    )
    
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=1200,
        help="LLM API request timeout in seconds (default: 1200)"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing state if available (default: true)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true", 
        help="Start fresh, ignore existing state"
    )
    
    parser.add_argument(
        "--cleanup", "-c",
        action="store_true",
        help="Clean up intermediate files when processing completes"
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show status of existing pipeline state and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def check_prerequisites():
    """Check if required dependencies and setup are available."""
    issues = []
    
    # Check for LLM API key if analysis is requested
    if not os.getenv("OPENAI_ACCESS_TOKEN"):
        issues.append(
            "⚠️  OPENAI_ACCESS_TOKEN not found. Content analysis will be skipped.\n"
            "   Set your API key: export OPENAI_ACCESS_TOKEN='your-api-key'"
        )
    
    # Check for ffmpeg command-line tool
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, check=False)
        if result.returncode != 0:
            issues.append("⚠️  ffmpeg command not found. Audio extraction will fail.")
    except FileNotFoundError:
        issues.append("⚠️  ffmpeg command not found. Audio extraction will fail.")
    
    # Check for OpenCV (used by video processor)
    try:
        import cv2
    except ImportError:
        issues.append("⚠️  OpenCV (cv2) not installed. Video processing will fail.")
    
    # Check for MLX Whisper (used by ASR processor)
    try:
        import mlx_whisper
    except ImportError:
        issues.append("⚠️  mlx_whisper not installed. Transcription will fail.")
    
    return issues


def show_status(orchestrator: PipelineOrchestrator):
    """Show current pipeline status."""
    summary = orchestrator.get_state_summary()
    
    if not summary:
        print("📄 No existing pipeline state found.")
        return
    
    print("📊 PIPELINE STATUS")
    print("=" * 50)
    print(f"Video: {Path(summary['video_path']).name}")
    print(f"Output: {summary['output_dir']}")
    print(f"Started: {summary['started_at']}")
    print(f"Updated: {summary['updated_at']}")
    print(f"Analysis Language: {summary['analysis_language']}")
    
    if summary['content_type']:
        print(f"Content Type: {summary['content_type']}")
    
    print(f"\nCompleted Steps: {len(summary['completed_steps'])}")
    for step in summary['completed_steps']:
        print(f"  ✅ {step}")
    
    if summary['current_step']:
        print(f"\n🔄 Current Step: {summary['current_step']}")
    
    if summary['failed_step']:
        print(f"\n❌ Failed Step: {summary['failed_step']}")
        print(f"Error: {summary['error_message']}")
    
    print(f"\n📍 Resume Point: {summary['resume_point']}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Convert paths
    video_path = Path(args.video_path)
    output_dir = Path(args.output_dir)
    
    # Create orchestrator with LLM configuration
    orchestrator = PipelineOrchestrator(
        output_dir,
        llm_model=args.llm_model,
        llm_endpoint=args.llm_endpoint,
        llm_timeout=args.llm_timeout
    )
    
    # Show status if requested
    if args.status:
        show_status(orchestrator)
        return
    
    # Check prerequisites
    issues = check_prerequisites()
    if issues:
        print("🔍 PREREQUISITE CHECK")
        print("=" * 30)
        for issue in issues:
            print(issue)
        print()
    
    # Validate input file
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not video_path.is_file():
        print(f"❌ Error: Path is not a file: {video_path}")
        sys.exit(1)
    
    # Determine resume behavior
    resume = args.resume and not args.no_resume
    if args.no_resume:
        resume = False
    
    print("🎬 VIDEO PROCESSING PIPELINE")
    print("=" * 40)
    print(f"📹 Input: {video_path}")
    print(f"📁 Output: {output_dir}")
    print(f"🌐 Analysis Language: {args.analysis_language}")
    print(f"🤖 LLM Model: {args.llm_model}")
    print(f"📡 LLM Endpoint: {args.llm_endpoint}")
    print(f"⏱️ LLM Timeout: {args.llm_timeout}s")
    
    if args.content_type:
        print(f"📋 Content Type: {args.content_type}")
    else:
        print("📋 Content Type: Auto-detect")
    
    if resume:
        print("🔄 Resume: Enabled")
    else:
        print("🔄 Resume: Disabled (fresh start)")
    
    print()
    
    try:
        # Process video
        results = orchestrator.process_video(
            video_path=video_path,
            analysis_language=args.analysis_language,
            content_type=args.content_type,
            resume=resume,
            cleanup_intermediate=args.cleanup
        )
        
        # Show summary
        print("\n📋 PROCESSING SUMMARY")
        print("=" * 30)
        
        if results.get("transcription"):
            transcription = results["transcription"]
            print(f"📝 Transcription: {len(transcription['text'])} characters")
            if transcription.get("confidence"):
                print(f"   Confidence: {transcription['confidence']:.2f}")
        
        if results.get("analysis"):
            analysis = results["analysis"]
            print(f"🧠 Analysis: {analysis['content_type']}")
            print(f"   Credibility: {analysis['overall_credibility']}")
            print(f"   Conclusions: {len(analysis['conclusions'])}")
            print(f"   Language: {analysis['response_language']}")
        
        # Show output files
        print(f"\n📁 Results saved to: {output_dir}")
        result_files = list(output_dir.glob("*.json"))
        for file in result_files:
            print(f"   📄 {file.name}")
        
        if not args.cleanup:
            print("\n💡 Tip: Use --cleanup to remove intermediate files")
            print(f"   Or run: python scripts/process_video.py --status {output_dir}")
        
        print("\n🎉 Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user")
        print("💾 State has been saved. Use --resume to continue later.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print("💾 State has been saved. Check the error and use --resume to continue.")
        sys.exit(1)


if __name__ == "__main__":
    main()
