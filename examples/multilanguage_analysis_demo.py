"""
Demo script showing multi-language content analysis functionality.

This script demonstrates how to analyze content and get results in different languages.
"""

import os
from video_asr_summary.analysis import ContentType
from video_asr_summary.analysis.analyzer import DefaultContentAnalyzer
from video_asr_summary.analysis.classifier import KeywordBasedClassifier
from video_asr_summary.analysis.llm_client import OpenAICompatibleClient
from video_asr_summary.analysis.prompt_templates import DefaultPromptTemplateManager


def main():
    """Demonstrate multi-language analysis functionality."""
    
    if not os.getenv("OPENAI_ACCESS_TOKEN"):
        print("⚠️  OPENAI_ACCESS_TOKEN not found!")
        print("Please set your API key to run this demo.")
        return
    
    print("=== MULTI-LANGUAGE CONTENT ANALYSIS DEMO ===\n")
    
    # Initialize components
    llm_client = OpenAICompatibleClient()
    template_manager = DefaultPromptTemplateManager()
    classifier = KeywordBasedClassifier()
    analyzer = DefaultContentAnalyzer(llm_client, template_manager, classifier)
    
    # Sample political content for analysis
    sample_text = (
        "The government's new economic policy has generated significant debate. "
        "Supporters claim it will boost employment and reduce inflation by 2% "
        "within the next year, citing economic models from leading universities. "
        "However, critics argue that the policy will increase the national debt "
        "by $500 billion and may lead to higher taxes for middle-class families. "
        "Independent economists are divided, with some predicting positive growth "
        "while others warn of potential recession risks."
    )
    
    # Languages to test
    test_languages = [
        {"code": "en", "name": "English"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"}
    ]
    
    print("Analyzing the same content in different response languages:\n")
    print(f"Content: {sample_text[:100]}...\n")
    
    for lang in test_languages:
        print(f"--- Analysis in {lang['name']} ({lang['code']}) ---")
        
        try:
            # Analyze with specific response language
            result = analyzer.analyze(
                sample_text, 
                content_type=ContentType.POLITICAL_COMMENTARY,
                response_language=lang['code']
            )
            
            print(f"✓ Analysis completed in {result.processing_time_seconds:.2f}s")
            print(f"Response Language: {result.response_language}")
            print(f"Overall Credibility: {result.overall_credibility}")
            print()
            
            if result.conclusions:
                print("Main Conclusion:")
                conclusion = result.conclusions[0]
                print(f"Statement: {conclusion.statement}")
                print(f"Evidence Quality: {conclusion.evidence_quality}")
                
                if conclusion.supporting_arguments:
                    print("Supporting Arguments:")
                    for i, arg in enumerate(conclusion.supporting_arguments[:2], 1):
                        print(f"  {i}. {arg}")
                print()
            
            if result.key_insights:
                print("Key Insight:")
                print(f"• {result.key_insights[0]}")
                print()
            
        except Exception as e:
            print(f"✗ Error analyzing in {lang['name']}: {e}")
            print()
        
        print("-" * 60)
        print()
    
    # Demonstrate mixed language content analysis
    print("=== ANALYZING MIXED LANGUAGE CONTENT ===\n")
    
    mixed_content = (
        "Today's technology conference featured presentations about artificial "
        "intelligence and machine learning. The keynote speaker discussed "
        "革新的なアルゴリズム (innovative algorithms) and their impact on "
        "l'économie numérique (digital economy). Experts predict that these "
        "tecnologías emergentes will transform various industries."
    )
    
    print(f"Mixed Language Content: {mixed_content}")
    print()
    
    try:
        # Analyze mixed content with English response
        result = analyzer.analyze(
            mixed_content,
            response_language="en"
        )
        
        print(f"Detected Content Type: {result.content_type.value}")
        print(f"Response Language: {result.response_language}")
        print()
        
        if result.conclusions:
            print("Analysis Result:")
            for conclusion in result.conclusions:
                print(f"• {conclusion.statement}")
        
    except Exception as e:
        print(f"Error analyzing mixed content: {e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
