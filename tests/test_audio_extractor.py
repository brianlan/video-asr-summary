"""Tests for audio processing components."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from video_asr_summary.core import AudioData
from video_asr_summary.audio.extractor import AudioExtractor, FFmpegAudioExtractor


class TestFFmpegAudioExtractor:
    """Test the FFmpeg-based audio extractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FFmpegAudioExtractor()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_video_path = self.temp_dir / "test_video.mp4"
        self.test_audio_path = self.temp_dir / "test_audio.wav"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_audio_default_settings(self):
        """Test audio extraction with default settings."""
        # Create dummy input file
        self.test_video_path.touch()
        
        with patch('subprocess.run') as mock_run, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('wave.open') as mock_wave:
            
            mock_run.return_value.returncode = 0
            
            # Mock wave file
            mock_wave_obj = Mock()
            mock_wave.return_value.__enter__.return_value = mock_wave_obj
            mock_wave_obj.getframerate.return_value = 44100
            mock_wave_obj.getnchannels.return_value = 2
            mock_wave_obj.getnframes.return_value = 44100  # 1 second
            
            result = self.extractor.extract_audio(
                self.test_video_path, 
                self.test_audio_path
            )
            
            # Verify ffmpeg command
            expected_cmd = [
                'ffmpeg',
                '-i', str(self.test_video_path),
                '-acodec', 'pcm_s16le',
                '-ac', '2',  # stereo
                '-ar', '44100',  # sample rate
                '-y',  # overwrite
                str(self.test_audio_path)
            ]
            mock_run.assert_called_once_with(
                expected_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Verify result
            assert isinstance(result, AudioData)
            assert result.file_path == self.test_audio_path
            assert result.sample_rate == 44100
            assert result.channels == 2
            assert result.duration_seconds == 1.0
            assert result.format == "wav"
    
    def test_extract_audio_custom_settings(self):
        """Test audio extraction with custom settings."""
        # Create dummy input file
        self.test_video_path.touch()
        
        with patch('subprocess.run') as mock_run, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('wave.open') as mock_wave:
            
            mock_run.return_value.returncode = 0
            
            # Mock wave file
            mock_wave_obj = Mock()
            mock_wave.return_value.__enter__.return_value = mock_wave_obj
            mock_wave_obj.getframerate.return_value = 16000
            mock_wave_obj.getnchannels.return_value = 1
            mock_wave_obj.getnframes.return_value = 32000  # 2 seconds at 16kHz
            
            result = self.extractor.extract_audio(
                self.test_video_path,
                self.test_audio_path,
                sample_rate=16000,
                channels=1,  # mono for speech recognition
                format="wav"
            )
            
            # Verify ffmpeg command with custom settings
            expected_cmd = [
                'ffmpeg',
                '-i', str(self.test_video_path),
                '-acodec', 'pcm_s16le',
                '-ac', '1',  # mono
                '-ar', '16000',  # 16kHz sample rate
                '-y',
                str(self.test_audio_path)
            ]
            mock_run.assert_called_once_with(
                expected_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            assert result.sample_rate == 16000
            assert result.channels == 1
            assert result.duration_seconds == 2.0
    
    def test_extract_audio_with_time_range(self):
        """Test audio extraction from specific time range."""
        self.test_video_path.touch()
        
        with patch('subprocess.run') as mock_run, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('wave.open') as mock_wave:
            
            mock_run.return_value.returncode = 0
            
            mock_wave_obj = Mock()
            mock_wave.return_value.__enter__.return_value = mock_wave_obj
            mock_wave_obj.getframerate.return_value = 44100
            mock_wave_obj.getnchannels.return_value = 2
            mock_wave_obj.getnframes.return_value = 441000  # 10 seconds
            
            self.extractor.extract_audio(
                self.test_video_path,
                self.test_audio_path,
                start_time=30.0,  # Start at 30 seconds
                duration=10.0     # Extract 10 seconds
            )
            
            # Verify ffmpeg command includes time range
            expected_cmd = [
                'ffmpeg',
                '-i', str(self.test_video_path),
                '-ss', '30.0',      # Start time
                '-t', '10.0',       # Duration
                '-acodec', 'pcm_s16le',
                '-ac', '2',
                '-ar', '44100',
                '-y',
                str(self.test_audio_path)
            ]
            mock_run.assert_called_once_with(
                expected_cmd,
                capture_output=True,
                text=True,
                check=True
            )
    
    def test_extract_audio_file_not_found(self):
        """Test extraction with non-existent input file."""
        non_existent = self.temp_dir / "nonexistent.mp4"
        
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            self.extractor.extract_audio(non_existent, self.test_audio_path)
    
    def test_extract_audio_ffmpeg_not_found(self):
        """Test extraction when ffmpeg is not installed."""
        self.test_video_path.touch()
        
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="FFmpeg not found"):
                self.extractor.extract_audio(self.test_video_path, self.test_audio_path)
    
    def test_extract_audio_ffmpeg_fails(self):
        """Test extraction when ffmpeg command fails."""
        self.test_video_path.touch()
        
        with patch('subprocess.run') as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.CalledProcessError(
                1, 'ffmpeg', stderr='Invalid input file'
            )
            
            with pytest.raises(RuntimeError, match="FFmpeg failed"):
                self.extractor.extract_audio(self.test_video_path, self.test_audio_path)
    
    def test_extract_audio_output_not_created(self):
        """Test when ffmpeg succeeds but output file is not created."""
        self.test_video_path.touch()
        
        # Mock subprocess to succeed
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Mock Path.exists at the module level
            with patch('video_asr_summary.audio.extractor.Path') as mock_path_cls:
                # Create mock path instances
                mock_video_path = Mock()
                mock_video_path.exists.return_value = True
                mock_audio_path = Mock()
                mock_audio_path.exists.return_value = False  # Output file not created
                
                # Configure the Path constructor to return the appropriate mock
                def path_constructor(path_str):
                    if str(path_str) == str(self.test_video_path):
                        return mock_video_path
                    elif str(path_str) == str(self.test_audio_path):
                        return mock_audio_path
                    else:
                        # Return a real Path for other cases
                        return Path(path_str)
                
                mock_path_cls.side_effect = path_constructor
                
                with pytest.raises(RuntimeError, match="Audio extraction failed"):
                    self.extractor.extract_audio(self.test_video_path, self.test_audio_path)


class TestAudioExtractorInterface:
    """Test the audio extractor interface."""
    
    def test_abstract_interface(self):
        """Test that AudioExtractor is properly abstract."""
        with pytest.raises(TypeError):
            AudioExtractor()  # type: ignore # Should not be instantiable
