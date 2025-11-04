#from google.colab import userdata
import requests
import json
import google.generativeai as genai
import anthropic
from mllm_code.config.settings import EVALUATION_MODEL_NAME

class CaptionEvaluator:
    """
    A class for evaluating satellite image captions using LLM judges.
    Supports both Gemini and Anthropic models.
    """
    
    def __init__(self, gemini_api_key: str = None, anthropic_api_key: str = None):
        """
        Initialize the CaptionEvaluator with API keys.
        
        Args:
            gemini_api_key (str): API key for Google Gemini
            anthropic_api_key (str): API key for Anthropic Claude
        """
        self.gemini_api_key = gemini_api_key
        self.anthropic_api_key = anthropic_api_key
        
        # Configure Gemini if API key is provided
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
    
    def evaluate(self, caption: str, model: str, weights: dict = None, threshold: float = 3.5) -> dict:
        """
        Evaluates a caption and returns complete results including scores, reasoning, and decision.
        
        Args:
            caption (str): The caption to evaluate
            model (str): The model to use ("gemini" or "anthropic")
            weights (dict): Optional custom weights for decision making
            threshold (float): Threshold for accept/reject decision
            
        Returns:
            dict: Complete evaluation results with scores, reasoning, and decision
                  
        """
        # Validate model choice
        if model.lower() not in ["gemini", "anthropic"]:
            raise ValueError(
                f"Unsupported model: {model}. Please use 'gemini' or 'anthropic'."
            )
        
        # Check if API key is available for the chosen model
        if model.lower() == "gemini" and not self.gemini_api_key:
            raise ValueError(
                "Gemini API key not provided. Please set gemini_api_key during initialization."
            )
        elif model.lower() == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "Anthropic API key not provided. Please set anthropic_api_key during initialization."
            )
        
        # Generate the judge prompt
        judge_prompt = self._create_judge_prompt(caption)
        
        # Call the appropriate API
        if model.lower() == "gemini":
            evaluation_result = self._call_gemini_api(judge_prompt)
        else:
            evaluation_result = self._call_anthropic_api(judge_prompt)
        
        # Calculate decision (evaluation_result is always a dict here - exceptions are raised on failure)
        decision = self._calculate_decision(evaluation_result, weights, threshold)
        
        # Return complete results
        return {
            "success": True,
            "model": model,
            "caption": caption,
            "scores": evaluation_result,
            "decision": decision,
            "decision_text": "ACCEPT" if decision == 1 else "REJECT"
        }
    
    def _calculate_decision(self, evaluation_result: dict, weights: dict = None, threshold: float = 3.5) -> int:
        """Calculate binary decision (0-1) based on weighted average of 0-5 category scores."""

    # Default equal weights for 5 criteria
        if weights is None:
            weights = {
                "Environmental_Focus": 1/5,
                "Specificity_Terminology": 1/5,
                "Processes_Patterns": 1/5,
                "Adherence_to_Constraints": 1/5,
                "Conciseness": 1/5
            }

        # Validate all keys are present
        required_keys = list(weights.keys())
        missing = [k for k in required_keys if k not in evaluation_result]
        if missing:
            raise ValueError(f"Missing evaluation fields: {missing}")

        # Compute weighted average (since weights sum to 1)
        weighted_score = sum(evaluation_result[k] * weights[k] for k in required_keys)

        print(f"Weighted Score: {weighted_score:.2f} / 5 | Threshold: {threshold:.2f}\n")

        # Decision: 1 = accept / 0 = reject
        return int(weighted_score >= threshold)



    
    def _create_judge_prompt(self, caption: str) -> str:
        """
        Creates the judge prompt for caption evaluation.
        
        Args:
            caption (str): The caption to evaluate
            
        Returns:
            str: The formatted judge prompt
        """
        return f"""
You are an expert environmental analyst and a meticulous reviewer of satellite imagery captions. Your task is to critically evaluate a generated caption for its accuracy, environmental focus, specificity, and adherence to professional standards, based *solely on the text of the caption itself*.

CONTEXT:
The captions describe environmental conditions observed in satellite images. 
The genearl goal is to provide accurate descriptions of land cover conditions, including vegetation coverage, built-up areas with industrial and urban development.
The particular goal is to identify compromised environmental conditions, including slums, garbage dump sites, open pit mining and industrial waste sites.

You understand the environmental significance of spectral data and band operations such as:
- Normalized Difference Vegetation Index (NDVI)
- Normalized Difference Water Index (NDWI)
- Normalized Difference Built-up Index (NDBI)
- Normalized Burn Ratio (NBR)
- Ferrous Minerals Index (FMI)

When evaluating, consider if the caption:
- Sounds like a plausible, accurate statement.
- Contains any internal contradictions or obvious factual errors.
- Adheres to the specific terminology commensurate with an environmental analyst.

EVALUATION CRITERIA (Rate each on a scale of 1-5, where 1 is the lowest and 5 the highest score, **always provide a numeric score**):
* **1. Environmental Focus:**
    * 1: Primarily describes non-environmental objects or is entirely irrelevant.
    * 3: Describes some environmental aspects but also includes generic observarions (e.g., "buildings," "roads") without environmental context, or is too vague.
    * 5: Focuses exclusively on environmental conditions (land cover, water sources, vegetation health, pollution, degradation of the landscape through mining, clearcuting or idustrialization) and their significance.
* **2. Accuracy & Plausibility (Self-Contained):**
    * 1: Contains significant factual errors or clear textual hallucinations (describing things that are inconsistent within the caption itself).
    * 3: Mostly plausible but with minor inaccuracies or vague statements.
    * 5: Factually plausible, with no observable hallucinations or inconsistencies.
* **3. Specificity & Terminology:**
    * 1: Uses generic, vague language; lacks specific environmental or scientific terminology.
    * 3: Uses some environmental terminology but could be more specific or precise.
    * 5: Employs precise, specific environmental terminology (e.g., "riparian vegetation", "open pit mining", "urban heat island effect," "Normalized Difference Vegetation Index") where appropriate.
* **4. Description of Processes and Patterns:**
    * 1: Only lists static objects; no mention of patterns, changes, or processes.
    * 3: Mentions some patterns or changes, but not central to the description or lacks depth.
    * 5: Effectively describes spatial patterns, and underlying environmental processes.
* **5. Adherence to Constraints:**
    * 1: Significantly violates constraints (e.g., focuses heavily on non-environmental objects without environmental impact, uses subjective terms like "beautiful" or "impressive").
    * 3: Minor violations of constraints.
    * 5: Fully adheres to all constraints (no basic objects without environmental impact, no subjective terms, factual language, under 200 words).
* **6. Conciseness of Expression (1-5):**
    * 1: Excessively verbose or too short to be informative.
    * 3: Reasonable length but could be more concise.
    * 5: Concise and impactful without unnecessary jargon, within word limit as per the requirement.

INSTRUCTIONS:
1.  First, provide step-by-step reasoning for your scores for each of the six criteria. Be specific, referencing parts of the caption where applicable.
2.  Second, provide the final scores in a JSON object. **Ensure all scores are provided as numbers (e.g., 1, 3.5, 5.0).**
3.  Then calculate the "Overall_Content_Score" as the average of the first five criteria (Environmental_Focus, Accuracy_Plausibility, Specificity_Terminology, Processes_Patterns_Changes, Adherence_to_Constraints). The "Conciseness" score is separate.

INPUT FOR EVALUATION:
Generated Caption: "{caption}"

EVALUATION:
Reasoning:
"""

    def _call_gemini_api(self, judge_prompt: str) -> dict:
        """Call the Gemini API and return structured JSON output."""
        try:
            model = genai.GenerativeModel(EVALUATION_MODEL_NAME)

            response_schema = {
                "type": "object",
                "properties": {
                    "Environmental_Focus": {"type": "number"},
                    "Specificity_Terminology": {"type": "number"},
                    "Processes_Patterns": {"type": "number"},
                    "Adherence_to_Constraints": {"type": "number"},
                    "Conciseness": {"type": "number"},
                    "Reasoning": {"type": "string"}
                }
            }

            response = model.generate_content(
                judge_prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 20,
                    "candidate_count": 1,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema
                }
            )

            # ✅ Check content and parse safely
            text = getattr(response, "text", None)
            if not text:
                raise ValueError("Gemini API returned empty or invalid response")
            
            return self._parse_llm_response(text)

        except Exception as e:
            # Catch all exceptions including API errors and AttributeError from genai.errors
            raise RuntimeError(f"Unexpected error during Gemini evaluation: {e}") from e



    def _call_anthropic_api(self, judge_prompt: str) -> dict:
        """Helper function to call Anthropic API using the Anthropic client library"""
        try:
            # Create the Anthropic client
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            # Define the response schema for structured output
            response_schema = {
                "type": "object",
                "properties": {
                    "Environmental_Focus": {"type": "number"},
                    "Specificity_Terminology": {"type": "number"},
                    "Processes_Patterns": {"type": "number"},
                    "Adherence_to_Constraints": {"type": "number"},
                    "Conciseness": {"type": "number"},
                    "Reasoning": {"type": "string"}
                }
            }
            
            # Generate message with structured output
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1,  # Low temperature for consistent evaluation
                messages=[
                    {
                        "role": "user",
                        "content": judge_prompt
                    }
                ],
                response_format={
                    "type": "json_object",
                    "schema": response_schema
                }
            )
            
            # Extract the JSON response
            if response.content and len(response.content) > 0:
                json_string = response.content[0].text

                return self._parse_llm_response(json_string)
            else:
                raise RuntimeError("Anthropic API returned empty response")
                
        except Exception as e:
            raise RuntimeError(f"Anthropic API request failed: {e}") from e



    def _parse_llm_response(self, json_string: str) -> dict:
        """Parse LLM response and extract numeric scores + reasoning text."""

        try:
            # Try direct JSON parse first
            parsed_json = json.loads(json_string)
            reasoning_text = parsed_json.get("Reasoning", "")

        except json.JSONDecodeError:
            # Try to extract JSON portion if mixed text+JSON
            json_start = json_string.find('{')
            json_end = json_string.rfind('}')
            if json_start == -1 or json_end == -1 or json_end < json_start:
                raise RuntimeError("No valid JSON found in LLM response.")

            json_sub = json_string[json_start:json_end + 1]
            reasoning_text = json_string[:json_start].strip()

            try:
                parsed_json = json.loads(json_sub)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse extracted JSON: {e}") from e

            # Clean reasoning
            if reasoning_text.lower().startswith("reasoning:"):
                reasoning_text = reasoning_text[len("Reasoning:"):].strip()

            # Prefer reasoning field in parsed JSON if present
            reasoning_text = parsed_json.get("Reasoning", reasoning_text)

        # 3️⃣ Validate keys and cast to float
        try:
            return {
                "Environmental_Focus": float(parsed_json["Environmental_Focus"]),
                "Specificity_Terminology": float(parsed_json["Specificity_Terminology"]),
                "Processes_Patterns": float(parsed_json["Processes_Patterns"]),
                "Adherence_to_Constraints": float(parsed_json["Adherence_to_Constraints"]),
                "Conciseness": float(parsed_json["Conciseness"]),
                "Reasoning": reasoning_text,
            }
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"Invalid JSON structure or missing fields: {e}") from e

    