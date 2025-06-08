"""Audio extraction components."""

import subprocess
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from video_asr_summary.core import AudioData


class AudioExtractor(ABC):
    """Abstract base class for audio extraction."""
    
    @abstractmethod
    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        sample_rate: int = 44100,
        channels: int = 2,
        format: str = "wav",
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> AudioData:
        """Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path where extracted audio will be saved
            sample_rate: Audio sample rate in Hz (default: 44100)
            channels: Number of audio channels (1=mono, 2=stereo, default: 2)
            format: Output audio format (default: "wav")
            start_time: Start time in seconds (optional)
            duration: Duration in seconds (optional)
            
        Returns:
            AudioData object with extraction results
            
        Raises:
            FileNotFoundError: If input video file doesn't exist
            RuntimeError: If extraction fails
        """
        pass


class FFmpegAudioExtractor(AudioExtractor):
    """Audio extractor using FFmpeg."""
    
    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        sample_rate: int = 44100,
        channels: int = 2,
        format: str = "wav",
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> AudioData:
        """Extract audio from video using FFmpeg."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-i', str(video_path)]
        
        # Add time range options if specified
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])
        if duration is not None:
            cmd.extend(['-t', str(duration)])
        
        # Add audio encoding options
        cmd.extend([
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ac', str(channels),     # Channel count
            '-ar', str(sample_rate),  # Sample rate
            '-y',                     # Overwrite output file
            str(output_path)
        ])
        
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed to extract audio: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
        
        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError("Audio extraction failed - output file not created")
        
        # Get audio properties from the extracted file
        try:
            with wave.open(str(output_path), 'rb') as wav_file:
                actual_sample_rate = wav_file.getframerate()
                actual_channels = wav_file.getnchannels()
                frames = wav_file.getnframes()
                duration_seconds = frames / actual_sample_rate if actual_sample_rate > 0 else 0.0
                
                return AudioData(
                    file_path=output_path,
                    duration_seconds=duration_seconds,
                    sample_rate=actual_sample_rate,
                    channels=actual_channels,
                    format=format
                )
        except wave.Error as e:
            raise RuntimeError(f"Failed to read extracted audio file: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading audio file: {e}")


class AudioExtractorFactory:
    """Factory for creating audio extractors."""
    
    @staticmethod
    def create_extractor(extractor_type: str = "ffmpeg") -> AudioExtractor:
        """Create an audio extractor of the specified type.
        
        Args:
            extractor_type: Type of extractor to create ("ffmpeg")
            
        Returns:
            AudioExtractor instance
            
        Raises:
            ValueError: If extractor_type is not supported
        """
        if extractor_type.lower() == "ffmpeg":
            return FFmpegAudioExtractor()
        else:
            raise ValueError(f"Unsupported extractor type: {extractor_type}")


# Convenience function for common use cases
def extract_audio_for_speech_recognition(
    video_path: Path, 
    output_path: Path,
    extractor_type: str = "ffmpeg"
) -> AudioData:
    """Extract audio optimized for speech recognition.
    
    Uses 16kHz mono audio which is optimal for most ASR systems.
    
    Args:
        video_path: Path to input video
        output_path: Path for output audio
        extractor_type: Type of extractor to use
        
    Returns:
        AudioData with extracted audio info
    """
    extractor = AudioExtractorFactory.create_extractor(extractor_type)
    return extractor.extract_audio(
        video_path=video_path,
        output_path=output_path,
        sample_rate=16000,  # Optimal for speech recognition
        channels=1,         # Mono is sufficient for speech
        format="wav"
    )
