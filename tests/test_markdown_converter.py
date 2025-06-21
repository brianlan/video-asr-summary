"""Tests for markdown converter functionality."""

import json
from pathlib import Path
from unittest.mock import Mock, patch
from video_asr_summary.analysis.markdown_converter import MarkdownConverter


class TestMarkdownConverter:
    """Test the MarkdownConverter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = MarkdownConverter()
        self.sample_analysis = {
            "content_type": "personal_casual_talk",
            "overall_credibility": "medium",
            "response_language": "en",
            "processing_time_seconds": 244.41,
            "timestamp": "2025-06-21T16:02:21.868187",
            "conclusions": [
                {
                    "statement": "AI coding tools are becoming more powerful",
                    "confidence": 0.95,
                    "supporting_arguments": [
                        "User successfully built complex features in 30 minutes",
                        "Integration tasks completed in under an hour"
                    ],
                    "evidence_quality": "moderate",
                    "logical_issues": ["anecdotal reasoning"]
                }
            ],
            "key_insights": [
                "AI tools require disciplined human-in-the-loop processes",
                "Context management is as important as the AI model itself"
            ],
            "potential_biases": [
                "Power-User Bias: Conclusions based on intensive use cases",
                "Recency Bias: Recent positive experience may overshadow limitations"
            ],
            "factual_claims": [
                "The speaker's name is Chris and he builds productivity apps",
                "Anthropic recently introduced new pricing plans"
            ],
            "saved_at": "2025-06-21T16:02:21.868324"
        }
    
    def test_convert_analysis_to_markdown_basic_structure(self):
        """Test that convert_analysis_to_markdown returns properly structured markdown."""
        # When
        result = self.converter.convert_analysis_to_markdown(self.sample_analysis)
        
        # Then
        assert isinstance(result, str)
        assert "# Video Analysis Report" in result
        assert "## Summary" in result
        assert "## Conclusions" in result
        assert "## Key Insights" in result
        assert "## Potential Biases" in result
        assert "## Factual Claims" in result
        
    def test_convert_analysis_with_metadata(self):
        """Test that metadata is properly included in markdown."""
        # When
        result = self.converter.convert_analysis_to_markdown(self.sample_analysis)
        
        # Then
        assert "**Content Type:** personal_casual_talk" in result
        assert "**Overall Credibility:** medium" in result
        assert "**Analysis Language:** en" in result
        assert "**Processing Time:** 244.41 seconds" in result
        
    def test_convert_conclusions_with_confidence_and_evidence(self):
        """Test that conclusions are formatted with confidence scores and evidence."""
        # When
        result = self.converter.convert_analysis_to_markdown(self.sample_analysis)
        
        # Then
        assert "### Conclusion 1" in result
        assert "**Confidence:** 95%" in result
        assert "**Evidence Quality:** moderate" in result
        assert "AI coding tools are becoming more powerful" in result
        assert "- User successfully built complex features in 30 minutes" in result
        assert "- Integration tasks completed in under an hour" in result
        assert "**Logical Issues:** anecdotal reasoning" in result
        
    def test_convert_key_insights_as_list(self):
        """Test that key insights are formatted as a bulleted list."""
        # When
        result = self.converter.convert_analysis_to_markdown(self.sample_analysis)
        
        # Then
        assert "- AI tools require disciplined human-in-the-loop processes" in result
        assert "- Context management is as important as the AI model itself" in result
        
    def test_convert_biases_with_detailed_formatting(self):
        """Test that biases are formatted with proper structure."""
        # When
        result = self.converter.convert_analysis_to_markdown(self.sample_analysis)
        
        # Then
        assert "- **Power-User Bias:** Conclusions based on intensive use cases" in result
        assert "- **Recency Bias:** Recent positive experience may overshadow limitations" in result
        
    def test_convert_factual_claims_as_numbered_list(self):
        """Test that factual claims are formatted as a numbered list."""
        # When
        result = self.converter.convert_analysis_to_markdown(self.sample_analysis)
        
        # Then
        assert "1. The speaker's name is Chris and he builds productivity apps" in result
        assert "2. Anthropic recently introduced new pricing plans" in result
        
    def test_convert_empty_analysis_handles_gracefully(self):
        """Test that empty or minimal analysis data is handled gracefully."""
        # Given
        minimal_analysis = {
            "content_type": "unknown",
            "overall_credibility": "low",
            "conclusions": [],
            "key_insights": [],
            "potential_biases": [],
            "factual_claims": []
        }
        
        # When
        result = self.converter.convert_analysis_to_markdown(minimal_analysis)
        
        # Then
        assert "# Video Analysis Report" in result
        assert "No conclusions available" in result or "conclusions" in result.lower()
        
    def test_save_markdown_to_file(self):
        """Test that markdown can be saved to a file."""
        # Given
        output_path = Path("/tmp/test_analysis.md")
        
        # When
        self.converter.save_analysis_markdown(self.sample_analysis, output_path)
        
        # Then
        assert output_path.exists()
        content = output_path.read_text()
        assert "# Video Analysis Report" in content
        assert "AI coding tools are becoming more powerful" in content
        
        # Cleanup
        output_path.unlink()
        
    def test_save_markdown_creates_directory_if_not_exists(self):
        """Test that save_analysis_markdown creates parent directories."""
        # Given
        output_path = Path("/tmp/test_dir/analysis.md")
        if output_path.parent.exists():
            output_path.parent.rmdir()
        
        # When
        self.converter.save_analysis_markdown(self.sample_analysis, output_path)
        
        # Then
        assert output_path.exists()
        assert output_path.parent.exists()
        
        # Cleanup
        output_path.unlink()
        output_path.parent.rmdir()


class TestMarkdownConverterIntegration:
    """Test integration of markdown converter with existing analysis data."""
    
    def test_convert_real_analysis_file(self):
        """Test conversion of a real analysis.json file."""
        # Given
        converter = MarkdownConverter()
        analysis_path = Path("output/andrey_karpathy_software/analysis.json")
        
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis_data = json.load(f)
            
            # When
            result = converter.convert_analysis_to_markdown(analysis_data)
            
            # Then
            assert isinstance(result, str)
            assert len(result) > 100  # Should be substantial content
            assert "# Video Analysis Report" in result
            assert "Claude Code" in result or "AI" in result  # Should contain content from the analysis


class TestPipelineIntegration:
    """Test integration of markdown converter with pipeline orchestrator."""
    
    @patch('video_asr_summary.pipeline.orchestrator.MarkdownConverter')
    def test_orchestrator_calls_markdown_conversion(self, mock_converter_class):
        """Test that orchestrator includes markdown conversion step."""
        # Given
        from video_asr_summary.pipeline.orchestrator import PipelineOrchestrator
        from video_asr_summary.pipeline.state_manager import PipelineState
        
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        # Create orchestrator with temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PipelineOrchestrator(temp_dir)
            
            # Mock state
            state = Mock(spec=PipelineState)
            state.output_dir = temp_dir
            
            # Mock state manager methods
            orchestrator.state_manager.is_step_completed = Mock(return_value=False)
            orchestrator.state_manager.update_step = Mock()
            orchestrator.state_manager.complete_step = Mock()
            
            sample_analysis = {
                "content_type": "test",
                "conclusions": [],
                "key_insights": []
            }
            
            # When
            result = orchestrator._convert_analysis_to_markdown(state, sample_analysis)
            
            # Then
            mock_converter_class.assert_called_once()
            mock_converter.save_analysis_markdown.assert_called_once()
            assert result is not None
        
    def test_markdown_step_in_process_flow(self):
        """Test that markdown conversion is included in the main process flow."""
        # This test would verify that the markdown conversion step is called
        # as part of the complete pipeline process
        pass  # Will implement after adding to orchestrator
