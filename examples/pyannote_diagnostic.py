#!/usr/bin/env python3
"""
Diagnostic script to understand pyannote.audio model requirements and licensing.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_pyannote_models():
    """Check what models pyannote speaker diarization actually uses."""
    print("🔍 Investigating pyannote.audio model requirements...")

    try:
        import pyannote.audio

        print(f"✅ pyannote.audio version: {pyannote.audio.__version__}")

        # Try to load the pipeline and see what happens
        print("\n📋 Attempting to load speaker-diarization-3.1 pipeline...")

        from pyannote.audio import Pipeline

        # Check if we have a token
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("❌ No HUGGINGFACE_TOKEN found in environment")
            return False

        print(f"✅ Found HF token: {token[:8]}...")

        try:
            # This will show us what's actually required
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=token
            )
            print("✅ Pipeline loaded successfully!")

            # Check what models it contains
            print("\n🔧 Pipeline components:")
            if hasattr(pipeline, "_models"):
                for name, model in pipeline._models.items():
                    print(f"  - {name}: {model}")

            return True

        except Exception as e:
            print(f"❌ Pipeline loading failed: {e}")
            print(f"Error type: {type(e).__name__}")

            # Check if it's a specific license issue
            if (
                "gated" in str(e).lower()
                or "license" in str(e).lower()
                or "access" in str(e).lower()
            ):
                print("\n🔑 This appears to be a license/access issue.")
                print(
                    "The speaker-diarization-3.1 pipeline likely uses multiple models:"
                )
                print("  1. Speaker segmentation model (to find speech segments)")
                print("  2. Speaker embedding model (to identify speakers)")
                print("  3. Clustering/diarization logic")
                print("\nYou may need to accept licenses for ALL component models:")
                print(
                    "  - Go to: https://huggingface.co/pyannote/speaker-diarization-3.1"
                )
                print("  - Check which models it depends on")
                print("  - Accept licenses for each dependency")

            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def explain_segmentation_difference():
    """Explain the difference between Whisper and pyannote segmentation."""
    print("\n" + "=" * 60)
    print("📚 SEGMENTATION TYPES EXPLAINED")
    print("=" * 60)

    print(
        """
🎤 WHISPER SEGMENTATION (Content-based):
   Purpose: Split audio by SPEECH CONTENT
   Criteria: Pauses, sentence boundaries, topics
   Output:  "WHAT was said WHEN"
   
   Example:
   [0-30s]: "第一章,行为,The Behaviour,我们已经准备好..."
   [30-58s]: "在前几小时到几天内,是什么改变了神经系统..."

👥 PYANNOTE SEGMENTATION (Speaker-based):  
   Purpose: Split audio by SPEAKER IDENTITY
   Criteria: Voice characteristics, acoustic features
   Output:  "WHO was speaking WHEN"
   
   Example:
   [0-25s]:  SPEAKER_00 (Professor)
   [25-45s]: SPEAKER_01 (Student) 
   [45-70s]: SPEAKER_00 (Professor)

🔗 INTEGRATION (The Magic):
   Combine both to get: "WHO said WHAT WHEN"
   
   Result:
   [0-25s]:  SPEAKER_00: "第一章,行为,The Behaviour..."
   [25-30s]: SPEAKER_01: "在前几小时到几天内..."
   [30-45s]: SPEAKER_01: "是什么改变了神经系统..."
"""
    )

    print("💡 Why we need pyannote's segmentation:")
    print("   - Whisper doesn't know about voice characteristics")
    print("   - pyannote doesn't understand speech content")
    print("   - Together they create speaker-attributed transcripts")


def test_with_simpler_model():
    """Try using a different, possibly non-gated model."""
    print("\n" + "=" * 60)
    print("🧪 TESTING ALTERNATIVE APPROACHES")
    print("=" * 60)

    try:
        from pyannote.audio import Pipeline

        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("❌ No token available for testing")
            return False

        # Try different model versions
        models_to_try = [
            "pyannote/speaker-diarization-3.0",  # Older version
            "pyannote/speaker-diarization",  # Default version
        ]

        for model_name in models_to_try:
            print(f"\n🔄 Trying {model_name}...")
            try:
                pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
                print(f"✅ {model_name} loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")

        return False

    except Exception as e:
        print(f"❌ Alternative testing failed: {e}")
        return False


def main():
    """Main diagnostic function."""
    print("🔬 pyannote.audio Model Diagnostic Tool")
    print("=" * 50)

    # Explain the conceptual difference first
    explain_segmentation_difference()

    # Check model requirements
    success = check_pyannote_models()

    if not success:
        print("\n🔄 Trying alternative models...")
        test_with_simpler_model()

    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(
        """
Your understanding is mostly correct! Here's the key insight:

✅ Whisper gives us CONTENT segmentation (what was said when)
✅ pyannote gives us SPEAKER segmentation (who was speaking when)  
✅ We need BOTH to create speaker-attributed transcripts

The segmentation-3.0 license you accepted might be for a component model.
The speaker-diarization-3.1 pipeline likely uses multiple models internally.

Next steps:
1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1  
2. Accept any additional license agreements
3. Check which component models need licenses
"""
    )


if __name__ == "__main__":
    main()
