import os
import json
import google.generativeai as genai

class CaptionEvaluator:
    """
    Evaluate satellite image captions across multiple metrics using Gemini LLM
    in a single API call.
    """

    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

        # Define all metrics
            self.metrics = {
            "Environmental_Impact": """ 
* **1. Environmental_Impact:**
    * 1: Primarily describes non-environmental objects and conditions or is entirely irrelevant.
    * 3: Describes some environmental impacts but also includes generic observations without relevance to environmental impacts or is too vague.
    * 5: Focuses exclusively on environmental impacts (compromised land cover, vulnerable water sources, threats to forests and green spaces, pollution, degradation of the landscape through mining, clearcutting or industrialization) and their significance.

""",
            "Accuracy_Plausibility": """
* ** Accuracy_Plausibility:**
    * 1: Contains significant factual errors or clear textual hallucinations (describing things that are inconsistent within the caption itself).
    * 3: Mostly plausible but with minor inaccuracies or vague statements.
    * 5: Scientifically precise, factually plausible, with no observable hallucinations or inconsistencies.

""",
            "Adherence_to_Constraints": """
* ** Adherence to Constraints:**
    * 1: Significantly violates constraints (e.g., focuses heavily on non-environmental objects without environmental impact, uses subjective terms like "beautiful" or "impressive").
    * 3: Minor violations of constraints.
    * 5: Fully adheres to all constraints (no basic objects without environmental impact, no subjective terms, factual language, under 200 words).

""",
 "Conciseness": """
* ** Conciseness:**
    * 1: Excessively verbose or too short to be informative.
    * 3: Reasonable length but could be more concise.
    * 5: Concise and impactful without unnecessary jargon, within word limit as per the requirement.
""",
      "Specificity_Terminology": """
* **  Specificity_Terminology:**
    * 1: Uses generic, vague language; lacks specific environmental or scientific terminology.
    * 3: Uses some environmental terminology but could be more specific or precise.
    * 5: Employs precise, specific environmental impact terminology (e.g.,  "open pit mining", "resource depletion", "habitat destruction", " ecosystem disruption", "urban heat island effect" ) where appropriate.
""",
 "Processes_Patterns": """
* ** Processes_Patterns:**
    * 1: Only lists static objects; no mention of patterns, changes, or processes.
    * 3: Mentions some patterns or changes, but not central to the description or lacks depth.
    * 5: Effectively describes spatial patterns and underlying environmental processes.
"""
        }


    def evaluate(self, caption: str, model: str = "gemini", weights: dict = None, threshold: float = 3.5) -> dict:
        """Evaluate caption across all metrics and print metric values"""
        if model.lower() != "gemini":
            return {"error": "Only Gemini model supported in this patch", "success": False}

        if not self.gemini_api_key:
            return {"error": "Gemini API key not provided", "success": False}

        # Generate prompt and call Gemini API
        prompt = self._create_judge_prompt(caption)
        response = self._call_gemini_api(prompt)

        if response is None:
            return {"error": "Failed to evaluate caption with Gemini", "success": False}

        # Compute decision
        decision = self._calculate_decision(response, weights, threshold)

        # Print metric values
        print("\n--- Metric-wise Scores ---")
        for metric in self.metrics.keys():
            score = response.get(metric, "N/A")
            print(f"{metric}: {score}")
        print(f"Decision: {'ACCEPT' if decision == 1 else 'REJECT'}")
        print(f"Reasoning: {response.get('Reasoning', '')}\n")

        # Return results dictionary
        return {
            "success": True,
            "model": model,
            "caption": caption,
            "scores": response,
            "decision": decision,
            "decision_text": "ACCEPT" if decision == 1 else "REJECT"
        }


    def _calculate_decision(self, evaluation_result: dict, weights: dict = None, threshold: float = 3.5) -> int:
        if not evaluation_result:
            return 0

        if weights is None:
            weights = {m: 1 for m in self.metrics.keys()}

        weighted_score = sum(evaluation_result.get(m, 0) * weights.get(m, 0) for m in self.metrics.keys()) \
                         / max(1, sum(weights.values()))
        return 1 if weighted_score >= threshold else 0

    def _create_judge_prompt(self, caption: str) -> str:
        criteria_text = "\n".join(self.metrics.values())
        return f"""
You are an expert environmental analyst and a meticulous reviewer of satellite imagery captions. Your task is to critically evaluate a generated caption for its scientific accuracy, environmental focus, specificity, and adherence to professional standards, based *solely on the text of the caption itself*.

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

CAPTION:
"{caption}"

EVALUATION CRITERIA (Rate 1–5):
{criteria_text}

Respond ONLY in valid JSON in this format:

{{
  "Environmental_Impact": <number>,
  "Accuracy_Plausibility": <number>,
  "Adherence_to_Constraints": <number>,
  "Conciseness": <number>,
  "Specificity_Terminology": <number>,
  "Processes_Patterns": <number>,
  "Reasoning": "<short explanation>"
}}
"""

    def _call_gemini_api(self, judge_prompt: str) -> dict:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')  # more stable variant

            response = model.generate_content(
                judge_prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 20
                }
            )

            if response.text:
                return self._parse_llm_response(response.text)
            return None

        except Exception as e:
            print("Gemini API request failed:", e)
            return None

    def _parse_llm_response(self, json_string: str) -> dict:
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            start, end = json_string.find("{"), json_string.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(json_string[start:end+1])
                except json.JSONDecodeError as e:
                    print("Error parsing extracted JSON:", e)
        return None
