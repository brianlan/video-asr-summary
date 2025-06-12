"""
Demo script showing LLM-based content analysis functionality.

This script demonstrates how to use the analysis module to analyze different types
of content and extract conclusions with supporting arguments.
"""

import os
from video_asr_summary.analysis.analyzer import DefaultContentAnalyzer
from video_asr_summary.analysis.classifier import KeywordBasedClassifier
from video_asr_summary.analysis.llm_client import OpenAICompatibleClient
from video_asr_summary.analysis.prompt_templates import DefaultPromptTemplateManager


def main():
    """Demonstrate content analysis functionality."""
    
    # Check if API key is available
    if not os.getenv("OPENAI_ACCESS_TOKEN"):
        print("Warning: OPENAI_ACCESS_TOKEN not found in environment variables.")
        print("Please set your API key to run actual LLM analysis.")
        print("Demo will show the analysis pipeline without actual LLM calls.\n")
        demo_without_llm()
        return
    
    # Initialize components
    print("Initializing content analysis components...")
    
    try:
        llm_client = OpenAICompatibleClient()
        template_manager = DefaultPromptTemplateManager()
        classifier = KeywordBasedClassifier()
        
        analyzer = DefaultContentAnalyzer(llm_client, template_manager, classifier)
        
        print("✓ Components initialized successfully\n")
        
        # Demo with sample content
        demo_with_llm(analyzer)
        
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        print("Falling back to demo without LLM calls.\n")
        demo_without_llm()


def demo_without_llm():
    """Demo the analysis pipeline components without actual LLM calls."""
    
    print("=== CONTENT ANALYSIS DEMO (Without LLM) ===\n")
    
    # Initialize components (except LLM client)
    template_manager = DefaultPromptTemplateManager()
    classifier = KeywordBasedClassifier()
    
    # Sample content for different types
    sample_contents = {
        "Political Commentary": (
            "The recent election results demonstrate that voters are increasingly "
            "concerned about healthcare policies and tax reform. The winning candidate's "
            "platform promises significant changes to government spending and immigration "
            "policies, which could reshape the political landscape for years to come."
        ),
        "News Report": (
            "Breaking news: Officials have confirmed that the investigation into the "
            "incident is ongoing. According to sources close to the matter, the reporter "
            "will provide updates as new information becomes available. The spokesperson "
            "issued a statement earlier today addressing the developing situation."
        ),
        "Technical Review": (
            "In this comprehensive technical review, we analyze the performance "
            "characteristics of the new device. Our testing methodology includes "
            "benchmark comparisons, feature evaluation, and specification analysis. "
            "The results show significant improvements in processing speed and efficiency."
        )
    }
    
    # Demonstrate classification
    print("1. CONTENT CLASSIFICATION\n")
    
    for content_name, text in sample_contents.items():
        print(f"Content: {content_name}")
        print(f"Sample: {text[:100]}...")
        
        # Classify content
        predicted_type = classifier.classify(text)
        print(f"Predicted Type: {predicted_type.value}")
        
        # Show confidence scores
        confidence_scores = classifier.get_classification_confidence(text)
        print("Confidence Scores:")
        for content_type, score in confidence_scores.items():
            if score > 0:
                print(f"  {content_type.value}: {score:.3f}")
        
        print()
    
    # Demonstrate prompt templates
    print("2. PROMPT TEMPLATES\n")
    
    templates = template_manager.list_templates()
    for template in templates:
        print(f"Template: {template.name}")
        print(f"Content Type: {template.content_type.value}")
        print(f"Description: {template.description}")
        print(f"System Prompt (first 100 chars): {template.system_prompt[:100]}...")
        print()
    
    print("Demo completed! Set OPENAI_ACCESS_TOKEN to run actual LLM analysis.")


def demo_with_llm(analyzer: DefaultContentAnalyzer):
    """Demo with actual LLM analysis."""
    
    print("=== CONTENT ANALYSIS DEMO (With LLM) ===\n")
    
    # Sample political commentary for analysis
    sample_text = (
        "The recent healthcare reform proposal has sparked intense debate across "
        "the political spectrum. Supporters argue that the plan will reduce costs "
        "for middle-class families and improve access to essential services. They "
        "point to studies showing that similar reforms in other countries have led "
        "to better health outcomes and lower per-capita spending. However, critics "
        "contend that the proposal will increase government debt and reduce the "
        "quality of care. They cite concerns from medical professionals about "
        "potential resource constraints and longer wait times. The economic impact "
        "remains a key point of contention, with economists divided on whether the "
        "long-term savings will offset the initial implementation costs."
    )
    
    print("Analyzing sample political commentary...\n")
    print(f"Content: {sample_text[:200]}...\n")
    
    try:
        # Perform analysis
        print("Sending to LLM for analysis...")
        result = analyzer.analyze(sample_text)
        
        # Display results
        print(f"✓ Analysis completed in {result.processing_time_seconds:.2f} seconds\n")
        
        print("=== ANALYSIS RESULTS ===\n")
        
        print(f"Content Type: {result.content_type.value}")
        print(f"Overall Credibility: {result.overall_credibility}")
        print()
        
        print("CONCLUSIONS:")
        for i, conclusion in enumerate(result.conclusions, 1):
            print(f"{i}. {conclusion.statement}")
            print(f"   Confidence: {conclusion.confidence:.2f}")
            print(f"   Evidence Quality: {conclusion.evidence_quality}")
            if conclusion.supporting_arguments:
                print("   Supporting Arguments:")
                for arg in conclusion.supporting_arguments:
                    print(f"     • {arg}")
            if conclusion.logical_issues:
                print("   Logical Issues:")
                for issue in conclusion.logical_issues:
                    print(f"     • {issue}")
            print()
        
        if result.key_insights:
            print("KEY INSIGHTS:")
            for insight in result.key_insights:
                print(f"• {insight}")
            print()
        
        if result.potential_biases:
            print("POTENTIAL BIASES:")
            for bias in result.potential_biases:
                print(f"• {bias}")
            print()
        
        if result.factual_claims:
            print("FACTUAL CLAIMS FOR VERIFICATION:")
            for claim in result.factual_claims:
                print(f"• {claim}")
            print()
        
        # Show classification confidence for transparency
        confidence_scores = analyzer.get_classification_confidence(sample_text)
        print("CLASSIFICATION CONFIDENCE:")
        for content_type, score in confidence_scores.items():
            if score > 0:
                print(f"• {content_type.value}: {score:.3f}")
        
    except Exception as e:
        print(f"Error during LLM analysis: {e}")
        print("This might be due to API issues or rate limiting.")


if __name__ == "__main__":
    main()
