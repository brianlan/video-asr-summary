"""OpenAI-compatible LLM client for content analysis."""

import json
import os
import time
from typing import Optional
import openai
from video_asr_summary.analysis import (
    AnalysisResult,
    Conclusion,
    ContentType,
    LLMClient,
    PromptTemplate
)
from datetime import datetime


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible LLM client using any OpenAI API compatible endpoint."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: str = "https://models.github.ai/inference",  
        model: str = "openai/gpt-4.1",
        timeout: int = 60
    ):
        """Initialize the OpenAI-compatible client.
        
        Args:
            api_key: API key. If None, will try to get from OPENAI_ACCESS_TOKEN env var
            base_url: Base URL for the API endpoint
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("OPENAI_ACCESS_TOKEN")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_ACCESS_TOKEN environment variable "
                "or pass api_key parameter."
            )
        
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        
        # Configure OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    
    def analyze(self, text: str, prompt_template: PromptTemplate, response_language: str = "en") -> AnalysisResult:
        """Analyze text using LLM with given prompt template.
        
        Args:
            text: Text to analyze
            prompt_template: Prompt template to use
            response_language: Language for the analysis response (ISO 639-1 code)
        """
        start_time = time.time()
        
        # Language mapping for more natural language specification
        language_names = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi"
        }
        
        language_name = language_names.get(response_language, "English")
        
        # Add language instruction to system prompt
        system_prompt_with_language = (
            f"{prompt_template.system_prompt}\n\n"
            f"IMPORTANT: Please provide your analysis response in {language_name}. "
            f"All text in your response should be in {language_name}, including "
            f"conclusions, insights, biases, and factual claims."
        )
        
        # Format the user prompt with the transcription
        user_prompt = prompt_template.user_prompt_template.format(
            transcription=text
        )
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt_with_language},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000,  # Reasonable limit for analysis
            )
            
            # Extract response content
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            
            # Parse the JSON response
            analysis_data = self._parse_llm_response(content)
            
            # Create analysis result
            processing_time = time.time() - start_time
            return self._create_analysis_result(
                analysis_data, 
                prompt_template.content_type,
                processing_time,
                response_language
            )
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e
    
    def _parse_llm_response(self, content: str) -> dict:
        """Parse LLM response content as JSON."""
        # Try to extract JSON from response if it contains additional text
        content = content.strip()
        
        # Look for JSON block if response contains markdown formatting
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
        
        # Find JSON object boundaries
        if content.startswith("{") and content.endswith("}"):
            return json.loads(content)
        else:
            # Try to find JSON object in the content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_content = content[start:end]
                return json.loads(json_content)
            else:
                raise json.JSONDecodeError("No valid JSON found in response", content, 0)
    
    def _create_analysis_result(
        self, 
        data: dict, 
        content_type: ContentType,
        processing_time: float,
        response_language: str = "en"
    ) -> AnalysisResult:
        """Create AnalysisResult from parsed JSON data."""
        
        # Parse conclusions
        conclusions = []
        for conclusion_data in data.get("conclusions", []):
            conclusion = Conclusion(
                statement=conclusion_data.get("statement", ""),
                confidence=float(conclusion_data.get("confidence", 0.0)),
                supporting_arguments=conclusion_data.get("supporting_arguments", []),
                evidence_quality=conclusion_data.get("evidence_quality", "insufficient"),
                logical_issues=conclusion_data.get("logical_issues", [])
            )
            conclusions.append(conclusion)
        
        return AnalysisResult(
            content_type=content_type,
            conclusions=conclusions,
            overall_credibility=data.get("overall_credibility", "low"),
            key_insights=data.get("key_insights", []),
            potential_biases=data.get("potential_biases", []),
            factual_claims=data.get("factual_claims", []),
            response_language=response_language,
            processing_time_seconds=processing_time,
            timestamp=datetime.now()
        )
