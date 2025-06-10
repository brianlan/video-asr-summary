"""Main content analyzer that coordinates classification and LLM analysis."""

from typing import Optional
from video_asr_summary.analysis import (
    AnalysisResult,
    ContentAnalyzer,
    ContentClassifier,
    ContentType,
    LLMClient,
    PromptTemplateManager
)


class DefaultContentAnalyzer(ContentAnalyzer):
    """Default implementation of content analyzer."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        template_manager: PromptTemplateManager,
        classifier: ContentClassifier
    ):
        """Initialize the content analyzer.
        
        Args:
            llm_client: LLM client for analysis
            template_manager: Prompt template manager
            classifier: Content classifier
        """
        self.llm_client = llm_client
        self.template_manager = template_manager
        self.classifier = classifier
    
    def analyze(
        self, 
        text: str, 
        content_type: Optional[ContentType] = None
    ) -> AnalysisResult:
        """Analyze content and extract conclusions with supporting arguments.
        
        Args:
            text: Text content to analyze
            content_type: Optional content type override. If None, will classify automatically
            
        Returns:
            AnalysisResult with conclusions and supporting analysis
        """
        # Determine content type
        if content_type is None:
            content_type = self.classifier.classify(text)
        
        # Get appropriate prompt template
        template = self.template_manager.get_template(content_type)
        
        # Perform LLM analysis
        return self.llm_client.analyze(text, template)
    
    def get_classification_confidence(self, text: str) -> dict:
        """Get classification confidence scores for debugging/transparency."""
        from video_asr_summary.analysis.classifier import KeywordBasedClassifier
        
        if isinstance(self.classifier, KeywordBasedClassifier):
            return self.classifier.get_classification_confidence(text)
        else:
            # Fallback for classifiers without confidence scoring
            classified_type = self.classifier.classify(text)
            return {classified_type: 1.0}
