"""Video processing components using OpenCV."""

import subprocess
import wave
from pathlib import Path

import cv2

from video_asr_summary.core import VideoInfo, AudioData, VideoProcessor


class OpenCVVideoProcessor(VideoProcessor):
    """Video processor using OpenCV for video operations."""
    
    def extract_info(self, video_path: Path) -> VideoInfo:
        """Extract information from video file using OpenCV."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration
            duration_seconds = frame_count / frame_rate if frame_rate > 0 else 0.0
            
            # Get file size
            file_size_bytes = video_path.stat().st_size
            
            return VideoInfo(
                file_path=video_path,
                duration_seconds=duration_seconds,
                frame_rate=frame_rate,
                width=width,
                height=height,
                file_size_bytes=file_size_bytes
            )
        finally:
            cap.release()
    
    def extract_audio(self, video_path: Path, output_path: Path) -> AudioData:
        """Extract audio from video file using ffmpeg."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-acodec', 'pcm_s16le',
            '-ac', '2',
            '-ar', '44100',
            '-y',  # Overwrite output file
            str(output_path)
        ]
        
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
        
        # Get audio properties
        if not output_path.exists():
            raise RuntimeError("Audio extraction failed - output file not created")
        
        try:
            with wave.open(str(output_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                frames = wav_file.getnframes()
                duration_seconds = frames / sample_rate if sample_rate > 0 else 0.0
                
                return AudioData(
                    file_path=output_path,
                    duration_seconds=duration_seconds,
                    sample_rate=sample_rate,
                    channels=channels,
                    format="wav"
                )
        except wave.Error as e:
            raise RuntimeError(f"Failed to read extracted audio file: {e}")
