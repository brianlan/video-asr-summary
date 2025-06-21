"""Test punctuation-aware segmentation in SpecializedASRIntegrator."""

from video_asr_summary.integration.specialized_asr_integrator import SpecializedASRIntegrator


class TestPunctuationAwareSegmentation:
    """Test the new punctuation-aware segmentation approach."""
    
    def test_find_punctuation_boundaries(self):
        """Test finding punctuation boundaries in text."""
        integrator = SpecializedASRIntegrator()
        
        punctuated_text = "你好世界，这是一个测试。今天天气很好！"
        boundaries = integrator._find_punctuation_boundaries(punctuated_text)
        
        # Should have boundaries at: start, after comma, after period, after exclamation, end
        assert len(boundaries) >= 3  # At least start, middle boundaries, end
        assert boundaries[0] == 0  # Start
        assert boundaries[-1] == len(punctuated_text)  # End
        
        # Check that punctuation positions are captured
        comma_found = any(abs(b - 6) <= 1 for b in boundaries)  # After 你好世界，
        period_found = any(abs(b - 13) <= 1 for b in boundaries)  # After 测试。
        
        assert comma_found, f"Comma boundary not found. Boundaries: {boundaries}"
        assert period_found, f"Period boundary not found. Boundaries: {boundaries}"
    
    def test_align_original_to_punctuated_text(self):
        """Test alignment between original and punctuated text."""
        integrator = SpecializedASRIntegrator()
        
        original_text = "你好世界这是一个测试"  # No punctuation
        punctuated_text = "你好世界，这是一个测试。"  # With punctuation
        
        mapping = integrator._align_original_to_punctuated_text(original_text, punctuated_text)
        
        # Check key mappings
        assert 0 in mapping  # First character
        assert mapping[0] == 0  # Should map to first position
        
        # Characters before punctuation should map correctly
        assert 3 in mapping  # 世 (4th character)
        assert mapping[3] == 3  # Should still be at position 3
        
        # Characters after comma should account for punctuation shift
        if 4 in mapping:  # 这 (5th character)
            assert mapping[4] >= 5  # Should be after the comma position
    
    def test_create_punctuation_aware_segments_integration(self):
        """Test the full punctuation-aware segmentation workflow."""
        integrator = SpecializedASRIntegrator()
        
        # Mock character-level segments (each character has timing)
        character_segments = [
            {"start": 0.0, "end": 0.2, "text": "你", "confidence": 0.9},
            {"start": 0.2, "end": 0.4, "text": "好", "confidence": 0.9},
            {"start": 0.4, "end": 0.6, "text": "世", "confidence": 0.9},
            {"start": 0.6, "end": 0.8, "text": "界", "confidence": 0.9},
            {"start": 0.8, "end": 1.0, "text": "这", "confidence": 0.9},
            {"start": 1.0, "end": 1.2, "text": "是", "confidence": 0.9},
            {"start": 1.2, "end": 1.4, "text": "测", "confidence": 0.9},
            {"start": 1.4, "end": 1.6, "text": "试", "confidence": 0.9},
        ]
        
        original_text = "你好世界这是测试"
        punctuated_text = "你好世界，这是测试。"
        
        result_segments = integrator._create_punctuation_aware_segments(
            punctuated_text, character_segments, original_text
        )
        
        # Should create segments based on punctuation boundaries
        assert len(result_segments) >= 2  # At least 2 segments due to comma and period
        
        # Check first segment (should end with comma)
        first_segment = result_segments[0]
        assert "，" in first_segment["text"]
        assert first_segment["start"] == 0.0  # Should start at beginning
        assert first_segment["end"] > first_segment["start"]  # Should have valid timing
        
        # Check that segments have proper text content
        all_text = "".join(seg["text"].strip("，。") for seg in result_segments)
        assert "你好世界" in all_text
        assert "这是测试" in all_text
    
    def test_punctuation_aware_vs_original_approach(self):
        """Demonstrate the improvement over the original approach."""
        integrator = SpecializedASRIntegrator()
        
        # Scenario: FunASR gives character-level timing, punctuation model adds boundaries
        original_text = "你好世界这是一个很长的测试句子"
        punctuated_text = "你好世界，这是一个很长的测试句子。"
        
        # Character-level segments (representing FunASR output)
        character_segments = []
        for i, char in enumerate(original_text):
            character_segments.append({
                "start": i * 0.1,
                "end": (i + 1) * 0.1,
                "text": char,
                "confidence": 0.9
            })
        
        result_segments = integrator._create_punctuation_aware_segments(
            punctuated_text, character_segments, original_text
        )
        
        # Verify that punctuation boundaries are respected
        assert len(result_segments) >= 2  # Should break on comma and period
        
        # Verify timing precision is maintained
        for segment in result_segments:
            assert segment["start"] >= 0.0
            assert segment["end"] > segment["start"]
            assert segment["end"] <= len(original_text) * 0.1  # Within expected range
        
        # Verify text content includes punctuation
        has_comma_segment = any("，" in seg["text"] for seg in result_segments)
        has_period_segment = any("。" in seg["text"] for seg in result_segments)
        
        assert has_comma_segment, "No segment contains comma punctuation"
        assert has_period_segment, "No segment contains period punctuation"
