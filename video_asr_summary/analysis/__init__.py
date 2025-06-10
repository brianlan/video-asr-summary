"""Analysis module for LLM-based content analysis with prompt templates."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from datetime import datetime


class ContentType(Enum):
    """Types of content for analysis."""
    
    POLITICAL_COMMENTARY = "political_commentary"
    NEWS_REPORT = "news_report"
    TECHNICAL_REVIEW = "technical_review"
    GENERAL = "general"


@dataclass
class Conclusion:
    """A conclusion identified in the content."""
    
    statement: str
    confidence: float  # 0.0 to 1.0
    supporting_arguments: List[str]
    evidence_quality: str  # "strong", "moderate", "weak", "insufficient"
    logical_issues: List[str]  # Any logical fallacies or issues identified


@dataclass
class AnalysisResult:
    """Result of LLM content analysis."""
    
    content_type: ContentType
    conclusions: List[Conclusion]
    overall_credibility: str  # "high", "medium", "low"
    key_insights: List[str]
    potential_biases: List[str]
    factual_claims: List[str]  # Claims that can be fact-checked
    processing_time_seconds: Optional[float] = None
    timestamp: Optional[datetime] = None


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    
    name: str
    content_type: ContentType
    system_prompt: str
    user_prompt_template: str  # Template with placeholders like {transcription}
    description: str
    version: str = "1.0"


class PromptTemplateManager(ABC):
    """Abstract base class for managing prompt templates."""
    
    @abstractmethod
    def get_template(self, content_type: ContentType) -> PromptTemplate:
        """Get prompt template for content type."""
        pass
    
    @abstractmethod
    def list_templates(self) -> List[PromptTemplate]:
        """List all available templates."""
        pass
    
    @abstractmethod
    def add_template(self, template: PromptTemplate) -> None:
        """Add a new template."""
        pass


class ContentClassifier(ABC):
    """Abstract base class for content classification."""
    
    @abstractmethod
    def classify(self, text: str) -> ContentType:
        """Classify content type from text."""
        pass


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def analyze(
        self, 
        text: str, 
        prompt_template: PromptTemplate
    ) -> AnalysisResult:
        """Analyze text using LLM with given prompt template."""
        pass


class ContentAnalyzer(ABC):
    """Abstract base class for content analysis."""
    
    @abstractmethod
    def analyze(
        self, 
        text: str, 
        content_type: Optional[ContentType] = None
    ) -> AnalysisResult:
        """Analyze content and extract conclusions with supporting arguments."""
        pass
