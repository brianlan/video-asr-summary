"""Tests for video processing components."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from video_asr_summary.core import VideoInfo, AudioData
from video_asr_summary.video.opencv_processor import OpenCVVideoProcessor


class TestOpenCVVideoProcessor:
    """Test the OpenCV-based video processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = OpenCVVideoProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_video_path = self.temp_dir / "test_video.mp4"
        self.test_audio_path = self.temp_dir / "test_audio.wav"
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cv2.VideoCapture')
    def test_extract_info_success(self, mock_video_capture):
        """Test successful video info extraction."""
        # Mock the OpenCV VideoCapture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            3: 1920,    # CV_CAP_PROP_FRAME_WIDTH
            4: 1080,    # CV_CAP_PROP_FRAME_HEIGHT
            5: 30.0,    # CV_CAP_PROP_FPS
            7: 3600,    # CV_CAP_PROP_FRAME_COUNT
        }[prop]
        
        # Mock file size
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 50000000
            
            # Create a dummy file for the test
            self.test_video_path.touch()
            
            video_info = self.processor.extract_info(self.test_video_path)
            
            assert isinstance(video_info, VideoInfo)
            assert video_info.file_path == self.test_video_path
            assert video_info.width == 1920
            assert video_info.height == 1080
            assert video_info.frame_rate == 30.0
            assert video_info.duration_seconds == 120.0  # 3600 frames / 30 fps
            assert video_info.file_size_bytes == 50000000
    
    @patch('cv2.VideoCapture')
    def test_extract_info_file_not_found(self, mock_video_capture):
        """Test video info extraction with non-existent file."""
        non_existent_path = self.temp_dir / "non_existent.mp4"
        
        with pytest.raises(FileNotFoundError):
            self.processor.extract_info(non_existent_path)
    
    @patch('cv2.VideoCapture')
    def test_extract_info_invalid_video(self, mock_video_capture):
        """Test video info extraction with invalid video file."""
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        # Create a dummy file
        self.test_video_path.touch()
        
        with pytest.raises(ValueError, match="Could not open video file"):
            self.processor.extract_info(self.test_video_path)
    
    @patch('cv2.VideoCapture')
    @patch('cv2.VideoWriter_fourcc')
    @patch('cv2.VideoWriter')
    def test_extract_audio_success(self, mock_video_writer, mock_fourcc, mock_video_capture):
        """Test successful audio extraction."""
        # Mock the OpenCV VideoCapture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, Mock()), (False, None)]  # One frame then end
        mock_cap.get.side_effect = lambda prop: {
            3: 1920,    # Width
            4: 1080,    # Height  
            5: 30.0,    # FPS
            7: 30,      # Frame count
        }[prop]
        
        # Mock VideoWriter
        mock_writer = Mock()
        mock_video_writer.return_value = mock_writer
        mock_fourcc.return_value = 123456
        
        # Create dummy input file
        self.test_video_path.touch()
        
        # Mock subprocess for ffmpeg
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # Mock audio file creation and properties
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('wave.open') as mock_wave:
                mock_wave_obj = Mock()
                mock_wave.return_value.__enter__.return_value = mock_wave_obj
                mock_wave_obj.getframerate.return_value = 44100
                mock_wave_obj.getnchannels.return_value = 2
                mock_wave_obj.getnframes.return_value = 44100  # 1 second of audio
                
                audio_data = self.processor.extract_audio(
                    self.test_video_path, 
                    self.test_audio_path
                )
                
                assert isinstance(audio_data, AudioData)
                assert audio_data.file_path == self.test_audio_path
                assert audio_data.sample_rate == 44100
                assert audio_data.channels == 2
                assert audio_data.duration_seconds == 1.0
                assert audio_data.format == "wav"
    
    def test_extract_audio_input_not_found(self):
        """Test audio extraction with non-existent input file."""
        non_existent_path = self.temp_dir / "non_existent.mp4"
        
        with pytest.raises(FileNotFoundError):
            self.processor.extract_audio(non_existent_path, self.test_audio_path)
