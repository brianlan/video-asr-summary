"""
Integration example showing how to add content analysis to the video processing pipeline.

This demonstrates how to analyze transcribed content for conclusions and supporting arguments.
"""

import os
from typing import Optional
from video_asr_summary.analysis import ContentType
from video_asr_summary.analysis.analyzer import DefaultContentAnalyzer
from video_asr_summary.analysis.classifier import KeywordBasedClassifier
from video_asr_summary.analysis.llm_client import OpenAICompatibleClient
from video_asr_summary.analysis.prompt_templates import DefaultPromptTemplateManager


def create_content_analyzer() -> DefaultContentAnalyzer:
    """Create and configure a content analyzer."""
    
    # Initialize components
    llm_client = OpenAICompatibleClient()
    template_manager = DefaultPromptTemplateManager()
    classifier = KeywordBasedClassifier()
    
    return DefaultContentAnalyzer(llm_client, template_manager, classifier)


def analyze_transcription_text(
    transcription_text: str, 
    content_type: Optional[ContentType] = None
) -> dict:
    """
    Analyze transcription text for conclusions and supporting arguments.
    
    Args:
        transcription_text: The transcribed text to analyze
        content_type: Optional content type override
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    
    if not os.getenv("OPENAI_ACCESS_TOKEN"):
        return {
            "error": "OPENAI_ACCESS_TOKEN not set",
            "message": "Please set your API key to perform content analysis"
        }
    
    try:
        # Create analyzer
        analyzer = create_content_analyzer()
        
        # Perform analysis
        result = analyzer.analyze(transcription_text, content_type)
        
        # Convert to dictionary for easy serialization
        return {
            "success": True,
            "content_type": result.content_type.value,
            "overall_credibility": result.overall_credibility,
            "processing_time_seconds": result.processing_time_seconds,
            "conclusions": [
                {
                    "statement": c.statement,
                    "confidence": c.confidence,
                    "supporting_arguments": c.supporting_arguments,
                    "evidence_quality": c.evidence_quality,
                    "logical_issues": c.logical_issues
                }
                for c in result.conclusions
            ],
            "key_insights": result.key_insights,
            "potential_biases": result.potential_biases,
            "factual_claims": result.factual_claims,
            "classification_confidence": analyzer.get_classification_confidence(transcription_text)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to analyze content"
        }


def demo_pipeline_integration():
    """Demonstrate how content analysis integrates with the video processing pipeline."""
    
    print("=== PIPELINE INTEGRATION DEMO ===\n")
    
    # Simulate different types of transcribed content
    test_cases = [
        {
            "title": "Political Commentary",
            "content": (
                "Today's political landscape shows clear evidence that voter turnout "
                "directly correlates with election outcomes. Recent studies from Harvard "
                "and MIT demonstrate that when turnout exceeds 65%, the results tend to "
                "favor more progressive candidates. However, critics argue that this "
                "analysis doesn't account for demographic shifts and gerrymandering effects. "
                "The data suggests we need comprehensive voting reform, but opponents claim "
                "such changes would be costly and potentially disenfranchise rural voters."
            ),
            "expected_type": ContentType.POLITICAL_COMMENTARY
        },
        {
            "title": "Technical Review",
            "content": (
                "Our comprehensive testing of the new smartphone processor shows remarkable "
                "improvements in both speed and efficiency. Benchmark results indicate a "
                "40% performance increase over the previous generation, with power consumption "
                "reduced by 25%. The new architecture handles machine learning tasks "
                "exceptionally well, though some users report occasional thermal throttling "
                "during intensive gaming sessions. Based on our methodology and test results, "
                "this processor represents a significant advancement in mobile computing."
            ),
            "expected_type": ContentType.TECHNICAL_REVIEW
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. ANALYZING: {test_case['title']}")
        print(f"Content preview: {test_case['content'][:100]}...")
        print()
        
        # Analyze content
        result = analyze_transcription_text(test_case['content'])
        
        if result.get("success"):
            print(f"✓ Analysis completed in {result['processing_time_seconds']:.2f}s")
            print(f"Detected Type: {result['content_type']}")
            print(f"Expected Type: {test_case['expected_type'].value}")
            print(f"Overall Credibility: {result['overall_credibility']}")
            print()
            
            # Show key findings
            if result['conclusions']:
                print("Key Conclusions:")
                for j, conclusion in enumerate(result['conclusions'], 1):
                    print(f"  {j}. {conclusion['statement']}")
                    print(f"     Evidence Quality: {conclusion['evidence_quality']}")
                    print(f"     Confidence: {conclusion['confidence']:.2f}")
                print()
            
            if result['key_insights']:
                print("Key Insights:")
                for insight in result['key_insights']:
                    print(f"  • {insight}")
                print()
            
            if result['potential_biases']:
                print("Potential Biases:")
                for bias in result['potential_biases']:
                    print(f"  • {bias}")
                print()
        
        else:
            print(f"✗ Analysis failed: {result.get('error', 'Unknown error')}")
            print(f"Message: {result.get('message', 'No additional information')}")
        
        print("-" * 80)
        print()


def main():
    """Main function to demonstrate pipeline integration."""
    
    print("Content Analysis Pipeline Integration")
    print("=" * 50)
    print()
    
    # Check API key
    if not os.getenv("OPENAI_ACCESS_TOKEN"):
        print("⚠️  OPENAI_ACCESS_TOKEN not found!")
        print("This demo requires an API key to run content analysis.")
        print("Please set the environment variable and try again.")
        print()
        print("Example usage after setting the API key:")
        print("export OPENAI_ACCESS_TOKEN='your-api-key'")
        print("python examples/pipeline_integration_demo.py")
        return
    
    # Run demo
    demo_pipeline_integration()
    
    print("Integration demo completed!")
    print()
    print("To integrate this into your pipeline:")
    print("1. Add content analysis after transcription")
    print("2. Use the analyze_transcription_text() function")
    print("3. Include the results in your PipelineResult")


if __name__ == "__main__":
    main()
