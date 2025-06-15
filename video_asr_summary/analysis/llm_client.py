"""OpenAI-compatible LLM client for content analysis."""

import json
import os
import time
from typing import Optional
import openai
import httpx
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
        base_url: str = "https://openai.newbotai.cn/v1",  
        # model: str = "gemini-2.5-pro-exp-03-25",
        model: str = "gemini-2.5-pro-exp-03-25",
        timeout: int = 300
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
        
        # Configure OpenAI client with httpx timeout
        http_client = httpx.Client(timeout=self.timeout)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client
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
            start_time = time.time()
            print(f"🧠 Making LLM API call for {response_language} analysis...")
            print(f"📝 Text length: {len(text)} characters")
            
            # Model-specific parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt_with_language},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,  # Low temperature for consistent analysis
            }
            
            # Adjust max_tokens based on model
            if "deepseek" in self.model.lower():
                api_params["max_tokens"] = 4000  # DeepSeek models need more tokens
            elif "gpt" in self.model.lower():
                api_params["max_tokens"] = 3000  # GPT models are efficient
            elif "qwen" in self.model.lower():
                api_params["max_tokens"] = 3500  # Qwen models need reasonable space
            else:
                api_params["max_tokens"] = 2500  # Default for other models
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            
            print(f"✅ LLM API call completed")
            
            # Extract response content with detailed debugging
            if not response.choices or len(response.choices) == 0:
                print(f"❌ No choices in LLM response: {response}")
                raise ValueError("No choices returned from LLM API")
                
            choice = response.choices[0]
            content = choice.message.content
            
            if not content or content.strip() == "":
                # Detailed debugging for empty responses
                print(f"❌ Empty response from LLM")
                print(f"   Response: {response}")
                print(f"   Choice: {choice}")
                if hasattr(choice, 'finish_reason'):
                    print(f"   Finish reason: {choice.finish_reason}")
                    if choice.finish_reason == "content_filter":
                        raise ValueError("LLM response was filtered due to content policy. Try with shorter or different content.")
                    elif choice.finish_reason == "length":
                        raise ValueError("LLM response was truncated. Try with shorter input text.")
                if hasattr(response, 'usage'):
                    print(f"   Token usage: {response.usage}")
                raise ValueError("Empty response from LLM - possible content filtering or API issues")
            
            print(f"📄 Response length: {len(content)} characters")
            
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
        # Check for empty content
        if not content or content.strip() == "":
            raise json.JSONDecodeError("Empty content provided to JSON parser", "", 0)
        
        # Try to extract JSON from response if it contains additional text
        content = content.strip()
        
        # Log content for debugging (truncated for privacy)
        content_preview = content[:200] + "..." if len(content) > 200 else content
        print(f"📄 Parsing LLM response (preview): {content_preview}")
        
        # Look for JSON block if response contains markdown formatting
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
            else:
                # No closing ```, extract from ```json to end
                content = content[start:].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end != -1:
                content = content[start:end].strip()
            else:
                # No closing ```, extract from ``` to end
                content = content[start:].strip()
                
        # If content doesn't start with {, try to find the JSON object
        if not content.startswith("{"):
            json_start = content.find("{")
            if json_start != -1:
                content = content[json_start:]
        
        # Find JSON object boundaries
        if content.startswith("{") and content.endswith("}"):
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed for content: {content[:500]}...")
                raise
        else:
            # Try to find JSON object in the content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_content = content[start:end]
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing failed for extracted content: {json_content[:500]}...")
                    # Try to fix incomplete JSON by finding the last complete field
                    try:
                        # Find the last complete field before the error
                        lines = json_content.split('\n')
                        valid_lines = []
                        brace_count = 0
                        bracket_count = 0
                        
                        for line in lines:
                            valid_lines.append(line)
                            brace_count += line.count('{') - line.count('}')
                            bracket_count += line.count('[') - line.count(']')
                            
                            # If we have balanced braces and brackets, try parsing
                            if brace_count == 0 and bracket_count == 0:
                                test_json = '\n'.join(valid_lines)
                                try:
                                    return json.loads(test_json)
                                except json.JSONDecodeError:
                                    continue
                        
                        # If that fails, try adding closing braces
                        incomplete_json = '\n'.join(valid_lines)
                        if brace_count > 0:
                            incomplete_json += '}' * brace_count
                        if bracket_count > 0:
                            incomplete_json += ']' * bracket_count
                            
                        print(f"🔧 Attempting to fix incomplete JSON...")
                        return json.loads(incomplete_json)
                        
                    except json.JSONDecodeError:
                        print(f"❌ Could not repair JSON. Original error: {e}")
                        raise
            else:
                print(f"❌ No JSON object found in content: {content[:500]}...")
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
