import json
from google import genai
import settings 


class CaptionEvaluator:
    """
    Evaluate satellite image captions across multiple metrics using Gemini LLM
    in a single API call.
    """

    def __init__(self, gemini_api_key: str = None):
        # Use key passed in OR from settings
        self.gemini_api_key = gemini_api_key or settings.GEMINI_API_KEY
        self.genai_client = None
        if self.gemini_api_key:
            self.genai_client = genai.Client(api_key=self.gemini_api_key)

        # Define metrics (unchanged)
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

    def evaluate(self, caption: str, model: str = None,
                 weights: dict = None, threshold: float = None) -> dict:
        """Evaluate caption across all metrics and return results"""
        model = model or settings.MODEL_NAME
        weights = weights or settings.DEFAULT_WEIGHTS
        threshold = threshold or settings.DEFAULT_THRESHOLD

        if not model.lower().startswith("gemini"):
            return {"error": "Only Gemini models supported", "success": False}

        if not self.gemini_api_key:
            return {"error": "Gemini API key not provided", "success": False}

        prompt = self._create_judge_prompt(caption)
        response = self._call_gemini_api(prompt)

        if response is None:
            return {"error": "Failed to evaluate caption with Gemini", "success": False}

        decision = self._calculate_decision(response, weights, threshold)

        # Print scores
        print("\n--- Metric-wise Scores ---")
        for metric in self.metrics.keys():
            print(f"{metric}: {response.get(metric, 'N/A')}")
        print(f"Decision: {'ACCEPT' if decision == 1 else 'REJECT'}")
        print(f"Reasoning: {response.get('Reasoning', '')}\n")

        return {
            "success": True,
            "model": model,
            "caption": caption,
            "scores": response,
            "decision": decision,
            "decision_text": "ACCEPT" if decision == 1 else "REJECT"
        }

    def _calculate_decision(self, evaluation_result: dict,
                            weights: dict, threshold: float) -> int:
        if not evaluation_result:
            return 0
        weighted_score = (
            sum(evaluation_result.get(m, 0) * weights.get(m, 0)
                for m in self.metrics.keys())
            / max(1, sum(weights.values()))
        )
        return 1 if weighted_score >= threshold else 0

    def _create_judge_prompt(self, caption: str) -> str:
        # (keep your long ideal_examples + criteria here as before)
        criteria_text = "\n".join(self.metrics.values())
        ideal_examples = """
Below are the ideal examples of captions that are rated 5:

HoucaoBauxiteDeposit_rgb_2024-10-29
Houcao Bauxite Deposit – Sentinel-2, October 29, 2024 (China)
The Sentinel-2 image of the Houcao Bauxite Deposit (October 29, 2024) shows an extensive open-pit operation in China. The reddish-brown exposed soil, characteristic of bauxite, indicates major land disturbance and potential soil erosion. A large tailings pond with unnaturally colored water suggests chemical contamination and risk of leakage into nearby water bodies and groundwater. Vegetation around the site is sparse and stressed, reflecting the environmental impact of bauxite mining.

BurnagaMine_rgb_2024-09-13
Burnaga Mine – Sentinel-2, September 13, 2024 (Rwanda)
This Sentinel-2 image from September 13, 2024 shows the Burnaga Mine in Rwanda, where aluminum- and iron-manganese-rich phosphate pegmatite is extracted. Extensive exposed brown soil and bare rock indicate heavy mining activity with potential soil erosion and water pollution. A tailings pond with unusually colored water signals chemical contamination and risks to surrounding water sources.

CrownestPass_rgb_2024-10-12
Crowsnest Pass Mines – Sentinel-2, October 12, 2024 (Canada)
The October 12, 2024 Sentinel-2 image captures the Crowsnest Pass mines in Canada, known for rare phosphate and silicate extraction. Large areas of deforestation, likely linked to a legacy of coal mining, expose bare soil and suggest soil erosion and habitat loss. Although no oil spills, wildfires, or algal blooms are visible, the image indicates increased sedimentation and groundwater contamination risk from ongoing mining activities.

KolweziMine_rgb_2024-10-11
Kolwezi Mine – Sentinel-2, October 11, 2024 (Democratic Republic of the Congo)
Captured on October 11, 2024, the Sentinel-2 image shows the Kolwezi Mine in the DRC, an old copper mine operating since 1903. A large open pit and adjacent tailings pond with unnaturally colored water indicate potential chemical contamination and risk of leakage into nearby water bodies. The surrounding area is largely devoid of vegetation and shows severe land degradation.

TsumebMine_rgb_2024-12-16
Tsumeb Mine – Sentinel-2, December 16, 2024 (Namibia)
The December 16, 2024 Sentinel-2 image depicts the Tsumeb Mine in Namibia, one of the world’s richest polymetallic sites, producing copper, silver, and gold from depths of over 1,000 meters. Extensive open-cast excavation and soil removal have caused significant land degradation. A large tailings pond with unusually colored water highlights potential chemical contamination.

RedLakeMines_rgb_2024-11-09
Red Lake Mines – Sentinel-2, November 9, 2024 (Canada)
This Sentinel-2 image from November 9, 2024 shows the Red Lake mines in Balmertown, Ontario, on the traditional lands of the Wabauskang and Lac Seul First Nations. Known for high-grade gold deposits, the site is under restoration. Nonetheless, a large tailings pond with unnaturally colored water signals ongoing environmental risk, with potential leakage into nearby water bodies and groundwater.

KingsMountainLithiumMine_rgb_2024-12-08
Kings Mountain Lithium Mine – Sentinel-2, December 8, 2024 (United States)
Captured on December 8, 2024, this Sentinel-2 image shows the open-pit Kings Mountain Mine in Cleveland County, North Carolina, one of the largest bedrock lithium deposits in the U.S. Significant land alteration suggests soil erosion and habitat loss. Although no direct evidence of water or air pollution is visible, ongoing mining may pose risks to nearby water sources and ecosystems.

FooteLithiumMine_rgb_2024-12-23
Foote Lithium Mine – Sentinel-2, November 12, 2024 (United States)
The November 12, 2024 Sentinel-2 image depicts the Foote Lithium Mine in North Carolina, operated by Albemarle Corporation. A large adjacent tailings pond with unnaturally colored water suggests chemical contamination. Extensive excavation is evident, likely causing land degradation and soil erosion. Although no oil spills are visible, the mine and tailings pond clearly indicate substantial environmental risks.

QuebradaBlancaMine_rgb_2024-12-28.png
Quebrada Blanca Mine – Sentinel-2, December 28, 2024 (Chile)
This Sentinel-2 image from December 28, 2024 shows the Quebrada Blanca mine in Tamarugal Province, Chile. A large tailings pond with unusually colored water signals potential chemical contamination. The image reveals widespread land degradation, with large areas of bare soil and disturbed terrain. Vegetation appears stressed or cleared, indicating habitat destruction from mining.

FimistonOpenPit_rgb_2024-12-25
Fimiston Open Pit – Sentinel-2, December 25, 2024 (Australia)
The December 25, 2024 Sentinel-2 image captures the Fimiston Open Pit, Australia’s largest open-cut gold mine. A large tailings pond adjacent to the mine contains water with an abnormal color, suggesting chemical contamination. Although the image does not directly show water quality or soil contamination, the tailings pond represents a potential environmental hazard to nearby communities.
"""

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

{ideal_examples}
        CAPTION: "{caption}"
        EVALUATION CRITERIA (Rate 1–5):
        {criteria_text}
        Respond ONLY in valid JSON ...
        """

    def _call_gemini_api(self, judge_prompt: str) -> dict:
        try:
            if not self.genai_client:
                return None
            response = self.genai_client.models.generate_content(
                model=settings.MODEL_NAME,
                contents=judge_prompt,
                config={
                    "temperature": settings.TEMPERATURE,
                    "top_p": settings.TOP_P,
                    "top_k": settings.TOP_K
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
                    return json.loads(json_string[start:end + 1])
                except json.JSONDecodeError as e:
                    print("Error parsing extracted JSON:", e)
        return None
