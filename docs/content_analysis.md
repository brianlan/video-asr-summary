# Content Analysis Feature

This module provides LLM-based content analysis to extract conclusions, supporting arguments, and assess credibility from transcribed audio/video content.

## Overview

The content analysis system is designed to help users understand:
- **What conclusions are claimed** in the content
- **What arguments or evidence support** these conclusions  
- **The quality and reliability** of the supporting evidence
- **Potential biases** in the presentation
- **Factual claims** that can be independently verified

## Features

- **Multiple Content Types**: Specialized analysis for Political Commentary, News Reports, Technical Reviews, and General content
- **Automatic Classification**: Intelligently detects content type using keyword-based classification
- **Customizable Prompts**: Easy to add new prompt templates for different scenarios
- **OpenAI Compatible**: Works with any OpenAI API-compatible endpoint (tested with GitHub Models)
- **Structured Output**: Consistent JSON format with conclusions, evidence quality, and bias detection
- **Integration Ready**: Designed to integrate seamlessly with existing video processing pipeline

## Quick Start

### 1. Set up API Access

```bash
export OPENAI_ACCESS_TOKEN="your-api-key"
```

### 2. Basic Usage

```python
from video_asr_summary.analysis.analyzer import DefaultContentAnalyzer
from video_asr_summary.analysis.classifier import KeywordBasedClassifier
from video_asr_summary.analysis.llm_client import OpenAICompatibleClient
from video_asr_summary.analysis.prompt_templates import DefaultPromptTemplateManager

# Initialize components
llm_client = OpenAICompatibleClient()
template_manager = DefaultPromptTemplateManager()
classifier = KeywordBasedClassifier()

analyzer = DefaultContentAnalyzer(llm_client, template_manager, classifier)

# Analyze content
text = "Your transcribed content here..."
result = analyzer.analyze(text)

# Access results
print(f"Content Type: {result.content_type}")
print(f"Overall Credibility: {result.overall_credibility}")

for conclusion in result.conclusions:
    print(f"Conclusion: {conclusion.statement}")
    print(f"Evidence Quality: {conclusion.evidence_quality}")
    print(f"Supporting Arguments: {conclusion.supporting_arguments}")
```

### 3. Pipeline Integration

```python
from examples.pipeline_integration_demo import analyze_transcription_text

# After getting transcription from your pipeline
transcription_result = your_asr_processor.transcribe(audio_path)
analysis_result = analyze_transcription_text(transcription_result.text)

if analysis_result["success"]:
    print("Analysis completed successfully!")
    print(f"Conclusions found: {len(analysis_result['conclusions'])}")
```

## Architecture

```
video_asr_summary/analysis/
├── __init__.py                 # Core interfaces and data models
├── analyzer.py                 # Main content analyzer
├── classifier.py               # Content type classification
├── llm_client.py              # OpenAI-compatible LLM client
└── prompt_templates.py        # Prompt templates for different content types
```

### Key Components

- **ContentAnalyzer**: Main interface for content analysis
- **PromptTemplateManager**: Manages prompt templates for different content types
- **ContentClassifier**: Automatically classifies content type
- **LLMClient**: Handles communication with LLM APIs

## Content Types

### Political Commentary
- Analyzes political arguments and policy discussions
- Focuses on evidence quality and potential political biases
- Identifies factual claims that can be fact-checked

### News Report  
- Evaluates journalistic quality and fact-based reporting
- Distinguishes between facts and editorial interpretations
- Assesses source quality and media bias

### Technical Review
- Assesses technical accuracy and methodology
- Evaluates test procedures and data quality
- Identifies potential commercial or selection biases

### General Content
- Fallback analysis for any content type
- General critical thinking evaluation
- Identifies logical fallacies and weak reasoning

## Output Format

The analysis returns an `AnalysisResult` with:

```python
{
    "content_type": "political_commentary",
    "overall_credibility": "medium",
    "conclusions": [
        {
            "statement": "Main conclusion text",
            "confidence": 0.85,
            "supporting_arguments": ["arg1", "arg2"],
            "evidence_quality": "moderate",
            "logical_issues": ["issue1", "issue2"]
        }
    ],
    "key_insights": ["insight1", "insight2"],
    "potential_biases": ["bias1", "bias2"],
    "factual_claims": ["claim1", "claim2"],
    "processing_time_seconds": 3.45
}
```

## Configuration

### Custom LLM Endpoint

```python
client = OpenAICompatibleClient(
    base_url="https://your-custom-endpoint.com",
    model="your-model-name",
    api_key="your-api-key"
)
```

### Custom Prompt Templates

```python
from video_asr_summary.analysis import ContentType, PromptTemplate

custom_template = PromptTemplate(
    name="Custom Analysis",
    content_type=ContentType.GENERAL,
    system_prompt="Your custom system prompt...",
    user_prompt_template="Analyze: {transcription}",
    description="Custom analysis template"
)

template_manager.add_template(custom_template)
```

### Custom Content Classification

```python
classifier = KeywordBasedClassifier()
classifier.add_keywords(ContentType.POLITICAL_COMMENTARY, ["democracy", "voting"])
```

## Testing

Run the comprehensive test suite:

```bash
PYTHONPATH=/path/to/project python -m pytest tests/test_analysis.py -v
```

## Examples

- `examples/content_analysis_demo.py`: Standalone analysis demonstration
- `examples/pipeline_integration_demo.py`: Integration with video processing pipeline

## API Reference

See the docstrings in each module for detailed API documentation:
- `video_asr_summary.analysis`: Core interfaces
- `video_asr_summary.analysis.analyzer`: Main analyzer class
- `video_asr_summary.analysis.llm_client`: LLM client implementation

## Multi-Language Support

The content analysis system supports responses in multiple languages, making it accessible to international users and enabling cross-language analysis scenarios.

### Supported Languages

- **English** (en) - Default
- **Spanish** (es) 
- **French** (fr)
- **German** (de)
- **Italian** (it)
- **Portuguese** (pt)
- **Russian** (ru)
- **Japanese** (ja)
- **Korean** (ko)
- **Chinese** (zh)
- **Arabic** (ar)
- **Hindi** (hi)

### Usage Examples

```python
# Analyze content with Spanish response
result = analyzer.analyze(text, response_language="es")

# Analyze English content but get results in Chinese
result = analyzer.analyze(english_text, response_language="zh")

# Pipeline integration with language preference
analysis_result = analyze_transcription_text(
    transcription_text, 
    response_language="fr"
)
```

### Use Cases

- **International Users**: Get analysis results in your preferred language
- **Cross-Language Analysis**: Analyze content in one language, get results in another
- **Consistent Output**: Ensure all analysis results are in the same language regardless of input
- **Educational Content**: Create analysis materials in specific languages

## Configuration
