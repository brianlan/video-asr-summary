#!/usr/bin/env python3
"""
Diagnostic script to test pyannote.audio setup and identify issues.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_pyannote_imports():
    """Test if pyannote.audio can be imported."""
    print("=== Testing pyannote.audio imports ===")

    try:
        import pyannote.audio

        print("✅ pyannote.audio imported successfully")
        print(f"   Version: {pyannote.audio.__version__}")

        from pyannote.audio import Pipeline

        print("✅ Pipeline class imported successfully")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_huggingface_token():
    """Test Hugging Face token availability."""
    print("\n=== Testing Hugging Face Token ===")

    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        print("✅ HUGGINGFACE_TOKEN environment variable found")
        print(f"   Token length: {len(token)} characters")
        print(f"   Token preview: {token[:10]}...")
        return token
    else:
        print("❌ HUGGINGFACE_TOKEN environment variable not found")
        print("   To get a token:")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a new token with 'Read' access")
        print("   3. Set it: export HUGGINGFACE_TOKEN=your_token_here")
        return None


def test_model_access(token):
    """Test if we can access the pyannote model."""
    print("\n=== Testing Model Access ===")

    if not token:
        print("❌ Cannot test model access without token")
        return False

    try:
        from pyannote.audio import Pipeline

        print("🔄 Attempting to load pyannote model...")
        print("   Model: pyannote/speaker-diarization-3.1")
        print("   This will download the model if not cached locally...")

        # This is where it might fail - downloading/loading the model
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )

        print("✅ Model loaded successfully!")
        print(f"   Pipeline type: {type(pipeline)}")
        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Common issues and solutions
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n💡 This looks like an authentication issue:")
            print("   - Check that your token is valid")
            print("   - Make sure you have 'Read' access")
            print("   - Try regenerating the token")
        elif "403" in str(e) or "Forbidden" in str(e):
            print("\n💡 This looks like a permissions issue:")
            print("   - You may need to accept the model license")
            print("   - Go to: https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("   - Click 'Agree and access repository'")
        elif "ConnectionError" in str(e) or "timeout" in str(e).lower():
            print("\n💡 This looks like a network issue:")
            print("   - Check your internet connection")
            print("   - Try again in a few minutes")
        else:
            print("\n💡 Unknown error - see full error above")

        return False


def test_torch_availability():
    """Test if PyTorch is available for pyannote."""
    print("\n=== Testing PyTorch Availability ===")

    try:
        import torch

        print("✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")

        # Check for CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  CUDA not available - will use CPU")

        # Test basic tensor operations
        x = torch.randn(3, 3)
        print("✅ Basic tensor operations work")

        return True

    except ImportError as e:
        print(f"❌ PyTorch import error: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch test error: {e}")
        return False


def test_audio_file_access():
    """Test if we can access the audio file."""
    print("\n=== Testing Audio File Access ===")

    audio_path = Path("/Users/rlan/Downloads/行为ch1.wav")

    if audio_path.exists():
        print(f"✅ Audio file found: {audio_path}")
        size_mb = audio_path.stat().st_size / 1024 / 1024
        print(f"   Size: {size_mb:.2f} MB")

        # Test if we can read it with soundfile (used by pyannote)
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            print(f"✅ Audio file readable:")
            print(f"   Duration: {info.duration:.2f} seconds")
            print(f"   Sample rate: {info.samplerate} Hz")
            print(f"   Channels: {info.channels}")
            print(f"   Format: {info.format}")
            return True
        except Exception as e:
            print(f"❌ Cannot read audio file: {e}")
            return False
    else:
        print(f"❌ Audio file not found: {audio_path}")
        return False


def main():
    """Run all diagnostic tests."""
    print("🔍 pyannote.audio Diagnostic Tool")
    print("=" * 50)

    # Run all tests
    imports_ok = test_pyannote_imports()
    token = test_huggingface_token()
    torch_ok = test_torch_availability()
    audio_ok = test_audio_file_access()

    if imports_ok and token:
        model_ok = test_model_access(token)
    else:
        model_ok = False
        print("\n⚠️  Skipping model test due to missing requirements")

    # Summary
    print("\n" + "=" * 50)
    print("🏁 DIAGNOSTIC SUMMARY")
    print("=" * 50)

    print(f"pyannote.audio imports: {'✅' if imports_ok else '❌'}")
    print(f"Hugging Face token:     {'✅' if token else '❌'}")
    print(f"PyTorch availability:   {'✅' if torch_ok else '❌'}")
    print(f"Audio file access:      {'✅' if audio_ok else '❌'}")
    print(f"Model loading:          {'✅' if model_ok else '❌'}")

    if all([imports_ok, token, torch_ok, audio_ok, model_ok]):
        print("\n🎉 All tests passed! pyannote.audio should work.")
        return True
    else:
        print("\n❌ Some tests failed. See details above for fixes.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
