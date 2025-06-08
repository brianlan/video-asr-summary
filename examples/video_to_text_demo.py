"""
Integration Demo: Video to Text Pipeline

This example demonstrates the complete flow from video to transcription,
integrating the audio extractor with the new ASR processor.
"""

import sys
from pathlib import Path
import tempfile

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_asr_summary.audio import FFmpegAudioExtractor
from video_asr_summary.asr import WhisperProcessor
from video_asr_summary.core import AudioData, TranscriptionResult


def demo_video_to_text_pipeline():
    """Demo the complete video-to-text pipeline."""
    print("=== Video to Text Pipeline Demo ===\n")
    
    # Step 1: Initialize components
    print("1. Initializing pipeline components...")
    audio_extractor = FFmpegAudioExtractor()
    asr_processor = WhisperProcessor(language="en")  # or "zh" for Chinese
    
    print(f"   Audio Extractor: {audio_extractor.__class__.__name__}")
    print(f"   ASR Processor: {asr_processor.__class__.__name__}")
    print(f"   ASR Model: {asr_processor.model_name}")
    print(f"   ASR Language: {asr_processor.language}")
    print()
    
    # Step 2: Simulate pipeline flow
    print("2. Pipeline Flow Overview:")
    print("   Video File → Audio Extraction → Speech Recognition → Text Output")
    print()
    
    # Step 3: Show what each step would do
    video_path = Path("/path/to/your/video.mp4")
    temp_audio_path = Path(tempfile.gettempdir()) / "extracted_audio.wav"
    
    print("3. Step-by-step process:")
    print(f"   Input: {video_path}")
    print(f"   Temp audio: {temp_audio_path}")
    print()
    
    print("   Step 3a: Audio Extraction (would extract audio from video)")
    print("   ```python")
    print("   audio_data = audio_extractor.extract_audio(")
    print(f"       video_path=Path('{video_path}'),")
    print(f"       output_path=Path('{temp_audio_path}'),")
    print("       sample_rate=16000,  # Optimized for speech")
    print("       channels=1,         # Mono for better ASR")
    print("       format='wav'        # Uncompressed for quality")
    print("   )")
    print("   ```")
    print()
    
    # Simulate what the audio data would look like
    simulated_audio_data = AudioData(
        file_path=temp_audio_path,
        duration_seconds=125.6,
        sample_rate=16000,
        channels=1,
        format="wav"
    )
    
    print("   Example audio data result:")
    print(f"     Duration: {simulated_audio_data.duration_seconds:.1f} seconds")
    print(f"     Sample rate: {simulated_audio_data.sample_rate} Hz")
    print(f"     Channels: {simulated_audio_data.channels}")
    print(f"     Format: {simulated_audio_data.format}")
    print()
    
    print("   Step 3b: Speech Recognition (would transcribe audio)")
    print("   ```python")
    print("   transcription = asr_processor.transcribe(audio_data.file_path)")
    print("   ```")
    print()
    
    # Simulate what the transcription would look like
    simulated_transcription = TranscriptionResult(
        text="Welcome to our presentation on artificial intelligence and machine learning. "
             "Today we'll cover the fundamentals of neural networks, deep learning architectures, "
             "and practical applications in computer vision and natural language processing.",
        confidence=0.92,
        segments=[
            {
                'start': 0.0,
                'end': 3.2,
                'text': 'Welcome to our presentation on artificial intelligence',
                'avg_logprob': -0.2,
                'no_speech_prob': 0.01
            },
            {
                'start': 3.2,
                'end': 6.8,
                'text': ' and machine learning.',
                'avg_logprob': -0.15,
                'no_speech_prob': 0.02
            },
            {
                'start': 6.8,
                'end': 12.5,
                'text': " Today we'll cover the fundamentals of neural networks,",
                'avg_logprob': -0.25,
                'no_speech_prob': 0.01
            },
            {
                'start': 12.5,
                'end': 16.3,
                'text': ' deep learning architectures,',
                'avg_logprob': -0.18,
                'no_speech_prob': 0.015
            },
            {
                'start': 16.3,
                'end': 22.1,
                'text': ' and practical applications in computer vision',
                'avg_logprob': -0.22,
                'no_speech_prob': 0.008
            },
            {
                'start': 22.1,
                'end': 25.4,
                'text': ' and natural language processing.',
                'avg_logprob': -0.19,
                'no_speech_prob': 0.005
            }
        ],
        language="en",
        processing_time_seconds=8.7
    )
    
    print("   Example transcription result:")
    print(f"     Text: {simulated_transcription.text}")
    print(f"     Language: {simulated_transcription.language}")
    print(f"     Confidence: {simulated_transcription.confidence:.2f}")
    print(f"     Processing time: {simulated_transcription.processing_time_seconds:.1f} seconds")
    print(f"     Segments: {len(simulated_transcription.segments)}")
    print()
    
    print("   Detailed segments with timestamps:")
    for i, segment in enumerate(simulated_transcription.segments):
        start = segment['start']
        end = segment['end']
        text = segment['text']
        confidence = 1.0 - segment['no_speech_prob']
        print(f"     {i+1}. [{start:5.1f}s - {end:5.1f}s] {text} (conf: {confidence:.3f})")
    print()
    
    print("4. Complete pipeline function:")
    print("   ```python")
    print("   def process_video_to_text(video_path: Path, language: str = 'en') -> TranscriptionResult:")
    print("       # Extract audio")
    print("       audio_extractor = FFmpegAudioExtractor()")
    print("       temp_audio = Path(tempfile.mkdtemp()) / 'audio.wav'")
    print("       audio_data = audio_extractor.extract_audio(")
    print("           video_path, temp_audio, sample_rate=16000, channels=1")
    print("       )")
    print("       ")
    print("       # Transcribe audio")
    print("       asr_processor = WhisperProcessor(language=language)")
    print("       transcription = asr_processor.transcribe(audio_data.file_path)")
    print("       ")
    print("       # Cleanup temp file")
    print("       temp_audio.unlink(missing_ok=True)")
    print("       ")
    print("       return transcription")
    print("   ```")
    print()
    
    print("5. Language support:")
    print("   - English: WhisperProcessor(language='en')")
    print("   - Chinese: WhisperProcessor(language='zh')")
    print("   - Auto-detect: WhisperProcessor()  # No language specified")
    print()
    
    print("6. Next steps for enhancement:")
    print("   - Speaker diarization: Identify different speakers in segments")
    print("   - Cross-validation: Use multiple ASR services and compare results")
    print("   - Chunked processing: Handle very long videos efficiently")
    print("   - Real-time processing: Stream audio for live transcription")
    print()
    
    print("=== Demo Complete ===")
    print("\nThe ASR processor is now ready for integration!")
    print("Use it with real video files by providing actual file paths.")


if __name__ == "__main__":
    demo_video_to_text_pipeline()
