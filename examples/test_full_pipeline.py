#!/usr/bin/env python3
"""
Test the full integrated pipeline: ASR + Diarization on Chinese audio.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from video_asr_summary.asr.whisper_processor import WhisperProcessor
from video_asr_summary.diarization.integrator import SegmentBasedIntegrator
from video_asr_summary.diarization.pyannote_processor import PyannoteAudioProcessor


def test_full_pipeline():
    """Test the complete ASR + Diarization pipeline."""
    print("🎬 Testing Full ASR + Diarization Pipeline")
    print("=" * 60)

    audio_file = "/Users/rlan/Downloads/行为ch1_fixed.wav"

    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False

    print(f"📁 Audio file: {audio_file}")
    print(f"📊 File size: {os.path.getsize(audio_file) / 1024 / 1024:.1f} MB")

    # Step 1: ASR with Whisper
    print("\n🎤 Step 1: Whisper ASR")
    print("-" * 30)
    try:
        whisper = WhisperProcessor(language="zh")
        asr_result = whisper.transcribe(Path(audio_file))

        print(f"✅ ASR completed: {len(asr_result.segments)} segments")
        print(f"   Total duration: {asr_result.segments[-1]['end']:.1f}s")
        avg_confidence = sum(
            seg.get("avg_logprob", 0) for seg in asr_result.segments
        ) / len(asr_result.segments)
        print(f"   Average confidence: {avg_confidence:.3f}")

        # Show first few ASR segments
        print("\n📝 First 3 ASR segments:")
        for i, seg in enumerate(asr_result.segments[:3]):
            print(
                f"   [{seg['start']:6.1f}s - {seg['end']:6.1f}s] {seg['text'][:50]}..."
            )

    except Exception as e:
        print(f"❌ ASR failed: {e}")
        return False

    # Step 2: Diarization with pyannote
    print("\n👥 Step 2: Speaker Diarization")
    print("-" * 30)
    try:
        auth_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        if not auth_token:
            print("❌ No Hugging Face token found")
            return False

        pyannote = PyannoteAudioProcessor(auth_token=auth_token, device="auto")
        diarization_result = pyannote.diarize(Path(audio_file))

        print(f"✅ Diarization completed: {diarization_result.num_speakers} speakers")
        print(f"   Speaker segments: {len(diarization_result.segments)}")

        # Show speaker summary
        speakers = set(s.speaker for s in diarization_result.segments)
        print("\n👥 Speaker breakdown:")
        for speaker in sorted(speakers):
            speaker_segments = [
                s for s in diarization_result.segments if s.speaker == speaker
            ]
            total_time = sum(s.end - s.start for s in speaker_segments)
            print(
                f"   {speaker}: {len(speaker_segments)} segments, {total_time:.1f}s total"
            )

    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return False

    # Step 3: Integration
    print("\n🔗 Step 3: ASR + Diarization Integration")
    print("-" * 30)
    try:
        integrator = SegmentBasedIntegrator()
        enhanced_result = integrator.integrate(asr_result, diarization_result)

        print(
            f"✅ Integration completed: {len(enhanced_result.speaker_attributed_segments)} enhanced segments"
        )

        # Show integrated results
        print("\n🎯 First 5 integrated segments (WHO said WHAT WHEN):")
        for i, seg in enumerate(enhanced_result.speaker_attributed_segments[:5]):
            speaker_text = f"[{seg.get('speaker', 'Unknown')}]"
            start_time = seg.get("start", 0)
            end_time = seg.get("end", 0)
            text = seg.get("text", "")
            confidence = seg.get("confidence", 0)

            print(f"   [{start_time:6.1f}s - {end_time:6.1f}s] {speaker_text}")
            print(f'      "{text[:60]}..." (conf: {confidence:.2f})')

        # Summary stats
        print("\n📊 Integration Summary:")
        total_segments = len(enhanced_result.speaker_attributed_segments)
        segments_with_speaker = len(
            [s for s in enhanced_result.speaker_attributed_segments if s.get("speaker")]
        )
        speaker_coverage = (
            segments_with_speaker / total_segments * 100 if total_segments > 0 else 0
        )

        print(f"   Total segments: {total_segments}")
        print(f"   Segments with speaker info: {segments_with_speaker}")
        print(f"   Speaker coverage: {speaker_coverage:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_pipeline()

    if success:
        print("\n🎉 Full pipeline test completed successfully!")
        print("\n✨ Key Achievements:")
        print("   • ✅ Whisper ASR: WHAT was said WHEN")
        print("   • ✅ pyannote diarization: WHO was speaking WHEN")
        print("   • ✅ Integration: WHO said WHAT WHEN")
        print("   • ✅ All processing done locally on your machine")
    else:
        print("\n❌ Full pipeline test failed!")
        sys.exit(1)
