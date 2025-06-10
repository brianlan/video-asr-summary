"""Prompt template management for different content types."""

from typing import Dict, List
from video_asr_summary.analysis import (
    ContentType, 
    PromptTemplate, 
    PromptTemplateManager
)


class DefaultPromptTemplateManager(PromptTemplateManager):
    """Default implementation of prompt template manager."""
    
    def __init__(self):
        self._templates: Dict[ContentType, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        
        # Political Commentary Template
        political_template = PromptTemplate(
            name="Political Commentary Analysis",
            content_type=ContentType.POLITICAL_COMMENTARY,
            system_prompt="""You are an expert political analyst tasked with objectively analyzing political commentary. Your goal is to identify claims, conclusions, and their supporting arguments while remaining politically neutral. Focus on the logical structure and evidence quality rather than agreeing or disagreeing with the positions.""",
            user_prompt_template="""Please analyze the following political commentary transcript and provide a structured analysis:

TRANSCRIPT:
{transcription}

Please provide your analysis in the following JSON format:
{{
    "conclusions": [
        {{
            "statement": "The main conclusion or claim being made",
            "confidence": 0.0-1.0,
            "supporting_arguments": ["argument 1", "argument 2"],
            "evidence_quality": "strong|moderate|weak|insufficient",
            "logical_issues": ["any logical fallacies or weak reasoning"]
        }}
    ],
    "overall_credibility": "high|medium|low",
    "key_insights": ["insight 1", "insight 2"],
    "potential_biases": ["bias 1", "bias 2"],
    "factual_claims": ["claim 1 that can be fact-checked", "claim 2"]
}}

Focus on:
1. What conclusions are being drawn
2. What evidence or reasoning supports each conclusion
3. The quality and reliability of the supporting evidence
4. Any logical fallacies or weak arguments
5. Potential political biases or one-sided presentation
6. Factual claims that can be independently verified""",
            description="Analyzes political commentary for conclusions, arguments, and credibility"
        )
        
        # News Report Template
        news_template = PromptTemplate(
            name="News Report Analysis",
            content_type=ContentType.NEWS_REPORT,
            system_prompt="""You are a media literacy expert analyzing news reports. Your goal is to evaluate the journalistic quality, fact-based reporting, and identify any editorial conclusions or interpretations presented as facts.""",
            user_prompt_template="""Please analyze the following news report transcript and provide a structured analysis:

TRANSCRIPT:
{transcription}

Please provide your analysis in the following JSON format:
{{
    "conclusions": [
        {{
            "statement": "The main conclusion or interpretation presented",
            "confidence": 0.0-1.0,
            "supporting_arguments": ["supporting fact 1", "supporting fact 2"],
            "evidence_quality": "strong|moderate|weak|insufficient",
            "logical_issues": ["any unsupported jumps in logic"]
        }}
    ],
    "overall_credibility": "high|medium|low",
    "key_insights": ["key finding 1", "key finding 2"],
    "potential_biases": ["editorial bias 1", "source bias 2"],
    "factual_claims": ["verifiable fact 1", "verifiable fact 2"]
}}

Focus on:
1. What facts are reported vs. what interpretations are made
2. How well conclusions are supported by the presented facts
3. The quality and variety of sources cited
4. Any editorial opinions presented as facts
5. Potential media bias or framing effects
6. Factual claims that can be independently verified""",
            description="Analyzes news reports for factual accuracy and editorial interpretation"
        )
        
        # Technical Review Template
        technical_template = PromptTemplate(
            name="Technical Review Analysis",
            content_type=ContentType.TECHNICAL_REVIEW,
            system_prompt="""You are a technical expert evaluating technical reviews and analyses. Your goal is to assess the technical accuracy, methodology, and validity of conclusions drawn from technical evidence.""",
            user_prompt_template="""Please analyze the following technical review transcript and provide a structured analysis:

TRANSCRIPT:
{transcription}

Please provide your analysis in the following JSON format:
{{
    "conclusions": [
        {{
            "statement": "The main technical conclusion or recommendation",
            "confidence": 0.0-1.0,
            "supporting_arguments": ["technical evidence 1", "test result 2"],
            "evidence_quality": "strong|moderate|weak|insufficient",
            "logical_issues": ["methodological issues", "unsupported claims"]
        }}
    ],
    "overall_credibility": "high|medium|low",
    "key_insights": ["technical insight 1", "insight 2"],
    "potential_biases": ["commercial bias", "selection bias"],
    "factual_claims": ["measurable claim 1", "testable claim 2"]
}}

Focus on:
1. What technical conclusions are drawn
2. What evidence, tests, or data support each conclusion
3. The quality and rigor of the testing methodology
4. Any unsupported technical claims or generalizations
5. Potential commercial or selection biases
6. Technical claims that can be independently tested or verified""",
            description="Analyzes technical reviews for methodology and conclusion validity"
        )
        
        # General Template
        general_template = PromptTemplate(
            name="General Content Analysis",
            content_type=ContentType.GENERAL,
            system_prompt="""You are an expert critical thinking analyst. Your goal is to identify the main arguments, conclusions, and evaluate the reasoning quality in any type of content.""",
            user_prompt_template="""Please analyze the following transcript and provide a structured analysis:

TRANSCRIPT:
{transcription}

Please provide your analysis in the following JSON format:
{{
    "conclusions": [
        {{
            "statement": "The main conclusion or claim being made",
            "confidence": 0.0-1.0,
            "supporting_arguments": ["argument 1", "argument 2"],
            "evidence_quality": "strong|moderate|weak|insufficient",
            "logical_issues": ["any logical issues or fallacies"]
        }}
    ],
    "overall_credibility": "high|medium|low",
    "key_insights": ["key insight 1", "key insight 2"],
    "potential_biases": ["potential bias 1", "potential bias 2"],
    "factual_claims": ["factual claim 1", "factual claim 2"]
}}

Focus on:
1. What main conclusions or claims are being made
2. What reasoning or evidence supports each conclusion
3. The strength and reliability of the supporting evidence
4. Any logical fallacies or weak reasoning patterns
5. Potential biases in presentation or argument selection
6. Factual claims that can be independently verified""",
            description="General analysis template for any content type"
        )
        
        self._templates[ContentType.POLITICAL_COMMENTARY] = political_template
        self._templates[ContentType.NEWS_REPORT] = news_template
        self._templates[ContentType.TECHNICAL_REVIEW] = technical_template
        self._templates[ContentType.GENERAL] = general_template
    
    def get_template(self, content_type: ContentType) -> PromptTemplate:
        """Get prompt template for content type."""
        if content_type not in self._templates:
            raise ValueError(f"No template found for content type: {content_type}")
        return self._templates[content_type]
    
    def list_templates(self) -> List[PromptTemplate]:
        """List all available templates."""
        return list(self._templates.values())
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add a new template."""
        self._templates[template.content_type] = template
