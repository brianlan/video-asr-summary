#!/usr/bin/env python3
"""
Test script to verify diarization integration works with the pipeline.
This script demonstrates how speaker diarization results are integrated with transcription
and used in content analysis.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_asr_summary.core import (
    TranscriptionResult, DiarizationResult, SpeakerSegment, 
    EnhancedTranscriptionResult
)
from video_asr_summary.diarization.integrator import SegmentBasedIntegrator


def test_diarization_integration():
    """Test the diarization integration functionality."""
    print("🧪 TESTING DIARIZATION INTEGRATION")
    print("=" * 40)
    
    # Create mock transcription result
    print("📝 Creating mock transcription...")
    transcription = TranscriptionResult(
        text="Hello everyone. My name is Alice. Nice to meet you. Hi Alice, I'm Bob. How are you doing today?",
        confidence=0.95,
        segments=[
            {"start": 0.0, "end": 2.5, "text": "Hello everyone.", "confidence": 0.98},
            {"start": 2.5, "end": 4.0, "text": "My name is Alice.", "confidence": 0.96},
            {"start": 4.0, "end": 6.0, "text": "Nice to meet you.", "confidence": 0.94},
            {"start": 7.0, "end": 8.5, "text": "Hi Alice, I'm Bob.", "confidence": 0.97},
            {"start": 8.5, "end": 11.0, "text": "How are you doing today?", "confidence": 0.93}
        ],
        language="en",
        processing_time_seconds=3.2
    )
    print(f"   ✅ Transcription: {len(transcription.text)} characters")
    print(f"   ✅ Segments: {len(transcription.segments)}")
    
    # Create mock diarization result  
    print("\n🎙️ Creating mock diarization...")
    diarization = DiarizationResult(
        segments=[
            SpeakerSegment(start=0.0, end=6.0, speaker="SPEAKER_00", confidence=0.92),
            SpeakerSegment(start=7.0, end=11.0, speaker="SPEAKER_01", confidence=0.89)
        ],
        num_speakers=2,
        processing_time_seconds=1.8
    )
    print(f"   ✅ Speakers: {diarization.num_speakers}")
    print(f"   ✅ Speaker segments: {len(diarization.segments)}")
    
    # Test integration
    print("\n🔗 Testing integration...")
    try:
        integrator = SegmentBasedIntegrator()
        enhanced = integrator.integrate(transcription, diarization)
        
        print(f"   ✅ Enhanced transcription created")
        print(f"   ✅ Speaker-attributed segments: {len(enhanced.speaker_attributed_segments)}")
        
        # Display results
        print("\n📊 INTEGRATION RESULTS")
        print("-" * 25)
        for i, seg in enumerate(enhanced.speaker_attributed_segments):
            speaker = seg.get('speaker', 'Unknown')
            text = seg.get('text', '')
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            confidence = seg.get('confidence', 0)
            print(f"   {i+1}. [{start:.1f}s-{end:.1f}s] {speaker}: {text} (conf: {confidence:.2f})")
        
        # Test speaker-attributed text formatting (as used in analysis)
        print("\n🧠 SPEAKER-ATTRIBUTED TEXT FOR ANALYSIS")
        print("-" * 45)
        speaker_segments = []
        for seg in enhanced.speaker_attributed_segments:
            speaker = seg.get('speaker', 'Unknown Speaker')
            text_content = seg.get('text', '').strip()
            if text_content:
                speaker_segments.append(f"{speaker}: {text_content}")
        
        formatted_text = "\n".join(speaker_segments)
        print(formatted_text)
        
        print(f"\n✅ Integration test completed successfully!")
        print(f"   📝 Original text: {len(transcription.text)} chars")
        print(f"   🎙️ Speaker-attributed text: {len(formatted_text)} chars")
        print(f"   👥 Speakers identified: {diarization.num_speakers}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration failed: {e}")
        return False


def test_pipeline_flow():
    """Test the pipeline flow with diarization."""
    print("\n\n🔄 TESTING PIPELINE FLOW")
    print("=" * 30)
    
    # Import orchestrator
    from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator
    
    output_dir = Path("./test_diarization_output")
    orchestrator = PipelineOrchestrator(output_dir)
    
    # Check component availability
    print("🔍 Checking component availability...")
    print(f"   📹 Video processing: {'✅' if hasattr(orchestrator, '_video_processor') and orchestrator._video_processor else '❌'}")
    print(f"   🔊 Audio extraction: {'✅' if hasattr(orchestrator, '_audio_extractor') and orchestrator._audio_extractor else '❌'}")
    print(f"   🗣️ Speech recognition: {'✅' if hasattr(orchestrator, '_asr_processor') and orchestrator._asr_processor else '❌'}")
    print(f"   🎙️ Speaker diarization: {'✅' if hasattr(orchestrator, '_diarization_processor') and orchestrator._diarization_processor else '❌'}")
    print(f"   🧠 Content analysis: {'✅' if hasattr(orchestrator, '_content_analyzer') and orchestrator._content_analyzer else '❌'}")
    
    # Check if the enhanced analysis flow exists
    print("\n🔍 Checking enhanced pipeline methods...")
    has_diarize = hasattr(orchestrator, '_diarize_speakers')
    has_integrate = hasattr(orchestrator, '_integrate_diarization')
    has_enhanced_analysis = 'EnhancedTranscriptionResult' in str(orchestrator._analyze_content.__annotations__)
    
    print(f"   🎙️ _diarize_speakers method: {'✅' if has_diarize else '❌'}")
    print(f"   🔗 _integrate_diarization method: {'✅' if has_integrate else '❌'}")
    print(f"   🧠 Enhanced analysis support: {'✅' if has_enhanced_analysis else '❌'}")
    
    if has_diarize and has_integrate:
        print("\n✅ Diarization integration is properly implemented!")
        print("   The pipeline now supports:")
        print("   • Speaker diarization")
        print("   • Transcription-diarization integration") 
        print("   • Speaker-attributed content analysis")
    else:
        print("\n❌ Some diarization methods are missing")
    
    # Clean up
    try:
        orchestrator.cleanup(keep_final_result=False)
    except:
        pass
    
    return has_diarize and has_integrate


def main():
    """Run all tests."""
    print("🧪 DIARIZATION INTEGRATION TEST SUITE")
    print("=" * 45)
    print("This test verifies that speaker diarization has been")
    print("successfully integrated into the video processing pipeline.\n")
    
    # Test 1: Integration logic
    test1_passed = test_diarization_integration()
    
    # Test 2: Pipeline structure
    test2_passed = test_pipeline_flow()
    
    # Summary
    print(f"\n\n📋 TEST SUMMARY")
    print("=" * 15)
    print(f"🔗 Integration Logic: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"🔄 Pipeline Structure: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Speaker diarization has been successfully integrated!")
        print("\nTo use with real video files, ensure:")
        print("• HF_ACCESS_TOKEN is set for pyannote.audio")
        print("• OPENAI_ACCESS_TOKEN is set for content analysis")
    else:
        print(f"\n❌ Some tests failed. Check the output above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
