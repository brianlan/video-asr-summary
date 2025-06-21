"""Test character alignment fixes in FunASR processor."""

from unittest.mock import patch
from video_asr_summary.asr.funasr_specialized_processor import FunASRSpecializedProcessor


class TestFunASRCharacterAlignment:
    """Test character alignment implementation in FunASR processor."""
    
    def test_split_text_to_characters(self):
        """Test character splitting from raw spaced text."""
        processor = FunASRSpecializedProcessor()
        
        # Test Chinese text with spaces
        raw_text = "你 好 世 界"
        char_pieces = processor._split_text_to_characters(raw_text)
        
        assert len(char_pieces) == 4
        assert char_pieces == ["你", "好", "世", "界"]
        
        # Test empty text
        assert processor._split_text_to_characters("") == []
        
        # Test mixed Chinese and English
        raw_text = "你 好 hello 世界"
        char_pieces = processor._split_text_to_characters(raw_text)
        assert char_pieces == ["你", "好", "hello", "世界"]
    
    def test_map_segments_to_characters(self):
        """Test mapping segments to character pieces."""
        processor = FunASRSpecializedProcessor()
        
        # Mock FunASR segments (timestamp format: [start_ms, end_ms])
        raw_segments = [
            [0, 500],      # 你: 0-0.5s  
            [500, 1000],   # 好: 0.5-1.0s
            [1000, 1500],  # 世: 1.0-1.5s
            [1500, 2000],  # 界: 1.5-2.0s
        ]
        char_pieces = ["你", "好", "世", "界"]
        
        character_segments = processor._map_segments_to_characters(raw_segments, char_pieces)
        
        assert len(character_segments) == 4
        
        # Verify first segment
        first_seg = character_segments[0]
        assert first_seg["text"] == "你"
        assert first_seg["start"] == 0.0
        assert first_seg["end"] == 0.5
        assert first_seg["is_character_level"] is True
        
        # Verify last segment
        last_seg = character_segments[3]
        assert last_seg["text"] == "界"
        assert last_seg["start"] == 1.5
        assert last_seg["end"] == 2.0
    
    def test_map_segments_mismatched_counts(self):
        """Test handling of mismatched segment and character counts."""
        processor = FunASRSpecializedProcessor()
        
        # More segments than characters
        raw_segments = [[0, 500], [500, 1000], [1000, 1500]]
        char_pieces = ["你", "好"]  # Only 2 characters
        
        character_segments = processor._map_segments_to_characters(raw_segments, char_pieces)
        
        # Should take minimum count
        assert len(character_segments) == 2
        assert character_segments[0]["text"] == "你"
        assert character_segments[1]["text"] == "好"
    
    def test_combine_character_segments(self):
        """Test combining character segments into larger text segments."""
        processor = FunASRSpecializedProcessor()
        
        # Create character-level segments
        character_segments = [
            {"start": 0.0, "end": 0.5, "text": "你", "confidence": 0.9},
            {"start": 0.5, "end": 1.0, "text": "好", "confidence": 0.9},
            {"start": 1.0, "end": 1.5, "text": "，", "confidence": 0.9},  # Punctuation
            {"start": 1.5, "end": 2.0, "text": "世", "confidence": 0.9},
            {"start": 2.0, "end": 2.5, "text": "界", "confidence": 0.9},
        ]
        
        final_text, final_segments = processor._combine_character_segments(character_segments)
        
        # Verify final text is properly combined
        assert final_text == "你好，世界"
        
        # Should break on punctuation, so expect at least 2 segments
        assert len(final_segments) >= 2
        
        # First segment should end at punctuation
        first_segment = final_segments[0]
        assert "，" in first_segment["text"]
        
        # Segments should have proper timing
        for segment in final_segments:
            assert segment["start"] < segment["end"]
            assert "id" in segment
    
    def test_combine_empty_character_segments(self):
        """Test combining empty character segments."""
        processor = FunASRSpecializedProcessor()
        
        final_text, final_segments = processor._combine_character_segments([])
        
        assert final_text == ""
        assert final_segments == []
    
    @patch('video_asr_summary.asr.funasr_specialized_processor.logger')
    def test_character_segment_logging(self, mock_logger):
        """Test that proper logging occurs during character processing."""
        processor = FunASRSpecializedProcessor()
        
        # Test with mismatched counts
        raw_segments = [[0, 500], [500, 1000], [1000, 1500]]
        char_pieces = ["你", "好"]
        
        processor._map_segments_to_characters(raw_segments, char_pieces)
        
        # Should log warning about mismatched counts
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Segment count" in warning_call
        assert "character count" in warning_call
    
    def test_full_integration_character_alignment(self):
        """Test the full character alignment workflow."""
        processor = FunASRSpecializedProcessor()
        
        # Test the methods work together
        raw_text = "你 好 世 界"
        char_pieces = processor._split_text_to_characters(raw_text)
        
        raw_segments = [
            [0, 500], [500, 1000], [1000, 1500], [1500, 2000]
        ]
        
        character_segments = processor._map_segments_to_characters(raw_segments, char_pieces)
        final_text, final_segments = processor._combine_character_segments(character_segments)
        
        # Final text should be properly combined without extra spaces
        assert final_text == "你好世界"
        
        # Should have proper segments with timing
        assert len(final_segments) > 0
        for segment in final_segments:
            assert segment["start"] >= 0.0
            assert segment["end"] > segment["start"]
            assert len(segment["text"]) > 0
