"""Tests for content analysis functionality."""

import json
import pytest
from unittest.mock import Mock, patch
from video_asr_summary.analysis import (
    AnalysisResult,
    ContentType,
    PromptTemplate
)
from video_asr_summary.analysis.analyzer import DefaultContentAnalyzer
from video_asr_summary.analysis.classifier import KeywordBasedClassifier
from video_asr_summary.analysis.llm_client import OpenAICompatibleClient
from video_asr_summary.analysis.prompt_templates import DefaultPromptTemplateManager


class TestKeywordBasedClassifier:
    """Tests for keyword-based content classifier."""
    
    def test_classify_political_content(self):
        """Test classification of political content."""
        classifier = KeywordBasedClassifier()
        
        political_text = (
            "The recent election results show that voters are concerned about "
            "government policies and healthcare reform. The candidate's position "
            "on taxes and immigration will be crucial for the campaign."
        )
        
        result = classifier.classify(political_text)
        assert result == ContentType.POLITICAL_COMMENTARY
    
    def test_classify_news_content(self):
        """Test classification of news content."""
        classifier = KeywordBasedClassifier()
        
        news_text = (
            "Breaking news: Officials have confirmed that the incident is under "
            "investigation. According to sources, the reporter will provide "
            "updates as the story develops."
        )
        
        result = classifier.classify(news_text)
        assert result == ContentType.NEWS_REPORT
    
    def test_classify_technical_content(self):
        """Test classification of technical content."""
        classifier = KeywordBasedClassifier()
        
        technical_text = (
            "In this technical review, we'll analyze the performance and "
            "specifications of the new device. Our testing methodology "
            "includes benchmarks and feature evaluation."
        )
        
        result = classifier.classify(technical_text)
        assert result == ContentType.TECHNICAL_REVIEW
    
    def test_classify_general_content(self):
        """Test classification of general content without specific keywords."""
        classifier = KeywordBasedClassifier()
        
        general_text = (
            "This is a general discussion about various topics without "
            "specific keywords that would classify it into other categories."
        )
        
        result = classifier.classify(general_text)
        assert result == ContentType.GENERAL
    
    def test_get_classification_confidence(self):
        """Test classification confidence scoring."""
        classifier = KeywordBasedClassifier()
        
        political_text = "The government's new policy on taxes will affect voters."
        confidence_scores = classifier.get_classification_confidence(political_text)
        
        assert isinstance(confidence_scores, dict)
        assert ContentType.POLITICAL_COMMENTARY in confidence_scores
        assert confidence_scores[ContentType.POLITICAL_COMMENTARY] > 0


class TestDefaultPromptTemplateManager:
    """Tests for prompt template manager."""
    
    def test_get_political_template(self):
        """Test getting political commentary template."""
        manager = DefaultPromptTemplateManager()
        template = manager.get_template(ContentType.POLITICAL_COMMENTARY)
        
        assert isinstance(template, PromptTemplate)
        assert template.content_type == ContentType.POLITICAL_COMMENTARY
        assert "political" in template.system_prompt.lower()
        assert "{transcription}" in template.user_prompt_template
    
    def test_get_news_template(self):
        """Test getting news report template."""
        manager = DefaultPromptTemplateManager()
        template = manager.get_template(ContentType.NEWS_REPORT)
        
        assert isinstance(template, PromptTemplate)
        assert template.content_type == ContentType.NEWS_REPORT
        assert "news" in template.system_prompt.lower()
    
    def test_get_technical_template(self):
        """Test getting technical review template."""
        manager = DefaultPromptTemplateManager()
        template = manager.get_template(ContentType.TECHNICAL_REVIEW)
        
        assert isinstance(template, PromptTemplate)  
        assert template.content_type == ContentType.TECHNICAL_REVIEW
        assert "technical" in template.system_prompt.lower()
    
    def test_get_general_template(self):
        """Test getting general template."""
        manager = DefaultPromptTemplateManager()
        template = manager.get_template(ContentType.GENERAL)
        
        assert isinstance(template, PromptTemplate)
        assert template.content_type == ContentType.GENERAL
    
    def test_list_templates(self):
        """Test listing all templates."""
        manager = DefaultPromptTemplateManager()
        templates = manager.list_templates()
        
        assert len(templates) == 4
        content_types = [t.content_type for t in templates]
        assert ContentType.POLITICAL_COMMENTARY in content_types
        assert ContentType.NEWS_REPORT in content_types
        assert ContentType.TECHNICAL_REVIEW in content_types
        assert ContentType.GENERAL in content_types
    
    def test_invalid_content_type(self):
        """Test error handling for invalid content type."""
        manager = DefaultPromptTemplateManager()
        
        # Test with a content type that doesn't exist in the manager
        # We'll manually create an invalid enum value
        class InvalidContentType:
            pass
        
        with pytest.raises(ValueError, match="No template found"):
            manager.get_template(InvalidContentType())  # type: ignore


class TestOpenAICompatibleClient:
    """Tests for OpenAI-compatible LLM client."""
    
    @patch.dict('os.environ', {'OPENAI_ACCESS_TOKEN': 'test-token'})
    def test_client_initialization(self):
        """Test client initialization with environment variable."""
        client = OpenAICompatibleClient()
        assert client.api_key == 'test-token'
        assert client.model == 'openai/gpt-4.1'
        assert client.base_url == 'https://models.github.ai/inference'
    
    def test_client_initialization_with_params(self):
        """Test client initialization with parameters."""
        client = OpenAICompatibleClient(
            api_key='custom-key',
            base_url='https://custom.api.com',
            model='custom-model'
        )
        assert client.api_key == 'custom-key'
        assert client.base_url == 'https://custom.api.com'
        assert client.model == 'custom-model'
    
    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                OpenAICompatibleClient()
    
    @patch('openai.OpenAI')
    def test_parse_llm_response_json(self, mock_openai):
        """Test parsing JSON response from LLM."""
        client = OpenAICompatibleClient(api_key='test-key')
        
        json_content = {
            "conclusions": [
                {
                    "statement": "Test conclusion",
                    "confidence": 0.8,
                    "supporting_arguments": ["arg1", "arg2"],
                    "evidence_quality": "strong",
                    "logical_issues": []
                }
            ],
            "overall_credibility": "high",
            "key_insights": ["insight1"],
            "potential_biases": ["bias1"],
            "factual_claims": ["claim1"]
        }
        
        json_string = json.dumps(json_content)
        result = client._parse_llm_response(json_string)
        assert result == json_content
    
    @patch('openai.OpenAI')
    def test_parse_llm_response_with_markdown(self, mock_openai):
        """Test parsing JSON response wrapped in markdown."""
        client = OpenAICompatibleClient(api_key='test-key')
        
        json_content = {"conclusions": [], "overall_credibility": "low"}
        markdown_response = f"Here's the analysis:\n\n```json\n{json.dumps(json_content)}\n```\n"
        
        result = client._parse_llm_response(markdown_response)
        assert result == json_content


class TestDefaultContentAnalyzer:
    """Tests for the main content analyzer."""
    
    def test_analyze_with_manual_content_type(self):
        """Test analysis with manually specified content type."""
        # Create mocks
        mock_llm = Mock()
        mock_template_manager = Mock()
        mock_classifier = Mock()
        
        # Setup mock returns
        mock_template = PromptTemplate(
            name="Test",
            content_type=ContentType.POLITICAL_COMMENTARY,
            system_prompt="Test system prompt",
            user_prompt_template="Test {transcription}",
            description="Test template"
        )
        mock_template_manager.get_template.return_value = mock_template
        
        expected_result = AnalysisResult(
            content_type=ContentType.POLITICAL_COMMENTARY,
            conclusions=[],
            overall_credibility="medium",
            key_insights=[],
            potential_biases=[],
            factual_claims=[]
        )
        mock_llm.analyze.return_value = expected_result
        
        # Create analyzer
        analyzer = DefaultContentAnalyzer(mock_llm, mock_template_manager, mock_classifier)
        
        # Test
        result = analyzer.analyze("test text", ContentType.POLITICAL_COMMENTARY)
        
        # Verify
        assert result == expected_result
        mock_template_manager.get_template.assert_called_once_with(ContentType.POLITICAL_COMMENTARY)
        mock_llm.analyze.assert_called_once_with("test text", mock_template)
        mock_classifier.classify.assert_not_called()  # Should not classify when type is provided
    
    def test_analyze_with_automatic_classification(self):
        """Test analysis with automatic content classification."""
        # Create mocks
        mock_llm = Mock()
        mock_template_manager = Mock()
        mock_classifier = Mock()
        
        # Setup mock returns
        mock_classifier.classify.return_value = ContentType.NEWS_REPORT
        
        mock_template = PromptTemplate(
            name="Test",
            content_type=ContentType.NEWS_REPORT,
            system_prompt="Test system prompt",
            user_prompt_template="Test {transcription}",
            description="Test template"
        )
        mock_template_manager.get_template.return_value = mock_template
        
        expected_result = AnalysisResult(
            content_type=ContentType.NEWS_REPORT,
            conclusions=[],
            overall_credibility="medium",
            key_insights=[],
            potential_biases=[],
            factual_claims=[]
        )
        mock_llm.analyze.return_value = expected_result
        
        # Create analyzer
        analyzer = DefaultContentAnalyzer(mock_llm, mock_template_manager, mock_classifier)
        
        # Test
        result = analyzer.analyze("test news text")
        
        # Verify
        assert result == expected_result
        mock_classifier.classify.assert_called_once_with("test news text")
        mock_template_manager.get_template.assert_called_once_with(ContentType.NEWS_REPORT)
        mock_llm.analyze.assert_called_once_with("test news text", mock_template)
