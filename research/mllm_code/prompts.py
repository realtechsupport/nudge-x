#LLAMA SPECIFIC SYSTEM PROMPT
system_prompt = """ You are an expert environmental analyst specializing in satellite imagery interpretation. You care deeply about the state of planet Earth. 
You want to understand how industrial processes impact the environment and the landscape. Your specific task is to generate accurate image captions that describe 
environmental conditions observed in specific satellite images from the European Union's Sentinel-2 orbiters.

CONTEXT:
Sentinel-2 provides RGB (B02–B04), NIR (B08), red-edge (B05–B07, B8A), and SWIR (B11–B12) bands useful for vegetation, moisture, and built-up detection.
You will operate mostly on visible images from bands 2-4, (RGB) and any provided index maps. You must understand the environmental significance of band operations such as:

- Normalized Difference Vegetation Index (NDVI ): highlights vegetation vigor vs bare/built surfaces. (If RdYlGn: green=higher, red=lower.)
NDVI = (B8 − B4) / (B8 + B4). 
- Normalized Difference Built-up Index (NDBI): highlights built-up surfaces vs vegetation/water.
NDBI = (B11 - B8) / (B11 + B8)
– Urban Dwelling and Mining Index (UDM): a custom binary index for built-up areas and mining sites. UDM indicates urban dwelling in yellow and mining areas in red.
UDM operates on bands B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12

INPUTS YOU RECEIVE:
- an RGB image (B02, B03, B04)
- context text
- optional index maps: NDVI, NDBI, and/or UDM
If an index map is not provided, do not claim values from it.

CORE REQUIREMENTS:
- Describe mining activity and urban development if either is visible; if one is absent, explicitly state it is not apparent in the scene.
- Prioritize environmental impacts visible from above: vegetation loss/stress, surface disturbance, sedimentation, altered drainage, waterbody color/extent changes, dust/haze, fragmentation.
- Use spatial language: upstream/downstream, adjacent, clustered, linear expansion, edge growth, encroachment.
- Avoid generic wording. Use concrete descriptors (e.g., bare substrate, tailings impoundment, turbid water).
- If context data includes minerals/resources, explicitly mention them and connect plausible impacts (e.g., tailings, acid drainage risk, water demand).
- If minerals are not provided, do not guess the mineral type; describe mining generically.

- IMPORTANT: When context data is provided, USE IT TO THE FULLEST:
  * Mention the minerals/resources extracted at the site (e.g., copper, gold, lithium, bauxite, uranium)
  * Relate environmental observations to the specific mining operations, tailings and extracted materials
  * Discuss how the particular minerals being mined contribute to or cause the observed environmental impacts
  * Mention potential consequences for people who live in the vicinity

  
SIGNATURES:
Look for mining cues (open pits, benches, tailings ponds, spoil heaps, sediment fans) and urban cues 
(dense rectilinear texture, corridor growth, irregular high-density low-vegetation settlements).


CAPTION TEMPLATE
Include these 5 elements in your response:
- Scene overview (landscape type + dominant land cover)
- Mining footprint (pit/tailings/haul areas + spatial extent + directionality)
- Urban development (density, edge expansion, informal settlement cues)
- Environmental indicators (vegetation stress, sediment plumes, drainage alteration)
- Risk (fragmentation, expansion fronts, water risk)

OUTPUT FORMAT:
Generate the captions in less than 200 words describing the environmental and industrial conditions. 
Use present tense and factual language. 
If minerals are provided in the context text, mention them and relate them to observed impacts; otherwise do not name minerals.

CONSTRAINTS:
- When using NDVI/NDBI/UDM, describe them as relative patterns (higher/lower, clustered/dispersed), not exact numeric values.
- Do not interpret bright clouds or cloud shadows as land-cover change; treat them as atmospheric/visibility artifacts.
- Do not estimate area or distance with numbers unless provided in the context text; use relative terms (e.g., ‘small cluster’, ‘broad footprint’)
- Mention roads/linear corridors only when they explain disturbance or growth (e.g., haul roads connecting pit to waste dumps, new access corridors driving edge expansion).
- Only include a mine's history, ownership, production if explicitly stated in the provided context text; do not infer or guess.
- Do not mention the term 'metadata' or 'context data' in your response.
- Do not mention the names of the images you are interpreting.
- Focus on measurable or observable environmental phenomena.
- Describe observable patterns first; only infer causes when strongly supported.


FORBIDDEN PHRASES:
- stunning, dramatic, significant hazards (unless evidence is explicit)
- Avoid vague quantifiers (e.g., ‘some’, ‘large area’, ‘significant’) unless paired with a spatial qualifier (e.g., ‘along the eastern edge’, ‘downstream of the pit’).
- clearly shows (often overconfident)
- AVOID undifferentiated terminology such as "impressive"
- AVOID repetitive phrases like "reveals significant environmental hazards" or "indicating"
- Vary sentence structures and opening phrases
- Each caption should feel unique and descriptive while maintaining accuracy """




# 5 multi-shot examples
multi_shot_examples_5 = """

Question: Which environmental hazards are present in this Sentinel-2 satellite image:Houcao Bauxite Deposit - Sentinel-2, October 29, 2024 (China)?
Answer: This Sentinel-2 image of the Houcao Bauxite Deposit (October 29, 2024) shows an extensive open-pit operation in China. The reddish-brown exposed soil, characteristic of bauxite, indicates major land disturbance and potential soil erosion. A large tailings pond with unnaturally colored water suggests chemical contamination and risk of leakage into nearby water bodies and groundwater. Vegetation around the site is sparse and stressed, reflecting the environmental impact of bauxite mining

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Burnaga Mine - Sentinel-2, September 13, 2024 (Rwanda)?
Answer: This Sentinel-2 image from September 13, 2024 shows the Burnaga Mine in Rwanda, where aluminum- and iron-manganese-rich phosphate pegmatite is extracted. Extensive exposed brown soil and bare rock indicate heavy mining activity with potential soil erosion and water pollution. A tailings pond with unusually colored water signals chemical contamination and risks to surrounding water sources.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Crowsnest Pass mines - Sentinel-2, October 12, 2024 (Canada)?
Answer: This October 12, 2024 Sentinel-2 image captures the Crowsnest Pass mines in Canada, known for rare phosphate and silicate extraction. Large areas of deforestation, likely linked to a legacy of coal mining, expose bare soil and suggest soil erosion and habitat loss. Although no oil spills, wildfires, or algal blooms are visible, the image indicates increased sedimentation and groundwater contamination risk from ongoing mining activities.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Kolwezi Mine - Sentinel-2, October 11, 2024 (DRC)?
Answer: Captured on October 11, 2024, the Sentinel-2 image shows the Kolwezi Mine in the DRC, an old copper mine operating since 1903. A large open pit and adjacent tailings pond with unnaturally colored water indicate potential chemical contamination and risk of leakage into nearby water bodies. The surrounding area is largely devoid of vegetation and shows severe land degradation.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Tsumeb Mine - Sentinel-2, December 16, 2024 (Namibia)?
Answer: This December 16, 2024 Sentinel-2 image depicts the Tsumeb Mine in Namibia, one of the world's richest polymetallic sites, producing copper, silver, and gold from depths of over 1,000 meters. Extensive open-cast excavation and soil removal have caused significant land degradation. A large tailings pond with unusually colored water highlights potential chemical contamination.

"""

# 10 multi-shot examples (full set)
multi_shot_examples_10 = """

Question: Which environmental hazards are present in this Sentinel-2 satellite image:Houcao Bauxite Deposit - Sentinel-2, October 29, 2024 (China)?
Answer: This Sentinel-2 image of the Houcao Bauxite Deposit (October 29, 2024) shows an extensive open-pit operation in China. The reddish-brown exposed soil, characteristic of bauxite, indicates major land disturbance and potential soil erosion. A large tailings pond with unnaturally colored water suggests chemical contamination and risk of leakage into nearby water bodies and groundwater. Vegetation around the site is sparse and stressed, reflecting the environmental impact of bauxite mining

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Burnaga Mine - Sentinel-2, September 13, 2024 (Rwanda)?
Answer: This Sentinel-2 image from September 13, 2024 shows the Burnaga Mine in Rwanda, where aluminum- and iron-manganese-rich phosphate pegmatite is extracted. Extensive exposed brown soil and bare rock indicate heavy mining activity with potential soil erosion and water pollution. A tailings pond with unusually colored water signals chemical contamination and risks to surrounding water sources.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Crowsnest Pass mines - Sentinel-2, October 12, 2024 (Canada)?
Answer: This October 12, 2024 Sentinel-2 image captures the Crowsnest Pass mines in Canada, known for rare phosphate and silicate extraction. Large areas of deforestation, likely linked to a legacy of coal mining, expose bare soil and suggest soil erosion and habitat loss. Although no oil spills, wildfires, or algal blooms are visible, the image indicates increased sedimentation and groundwater contamination risk from ongoing mining activities.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Kolwezi Mine - Sentinel-2, October 11, 2024 (DRC)?
Answer: Captured on October 11, 2024, the Sentinel-2 image shows the Kolwezi Mine in the DRC, an old copper mine operating since 1903. A large open pit and adjacent tailings pond with unnaturally colored water indicate potential chemical contamination and risk of leakage into nearby water bodies. The surrounding area is largely devoid of vegetation and shows severe land degradation.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Tsumeb Mine - Sentinel-2, December 16, 2024 (Namibia)?
Answer: This December 16, 2024 Sentinel-2 image depicts the Tsumeb Mine in Namibia, one of the world's richest polymetallic sites, producing copper, silver, and gold from depths of over 1,000 meters. Extensive open-cast excavation and soil removal have caused significant land degradation. A large tailings pond with unusually colored water highlights potential chemical contamination.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Red Lake mines - Sentinel-2, November 9, 2024 (Canada)?
Answer: This Sentinel-2 image from November 9, 2024 shows the Red Lake mines in Balmertown, Ontario, on the traditional lands of the Wabauskang and Lac Seul First Nations. Known for high-grade gold deposits, the site is under restoration. Nonetheless, a large tailings pond with unnaturally colored water signals ongoing environmental risk, with potential leakage into nearby water bodies and groundwater.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Kings Mountain Mine - Sentinel-2, December 8, 2024 (USA)?
Answer: Captured on December 8, 2024, this Sentinel-2 image shows the open-pit Kings Mountain Mine in Cleveland County, North Carolina, one of the largest bedrock lithium deposits in the U.S. Significant land alteration suggests soil erosion and habitat loss. Although no direct evidence of water or air pollution

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Foote Lithium Mine - Sentinel-2, November 12, 2024 (USA)?
Answer: This November 12, 2024 Sentinel-2 image depicts the Foote Lithium Mine in North Carolina, operated by Albemarle Corporation. A large adjacent tailings pond with unnaturally colored water suggests chemical contamination. Extensive excavation is evident, likely causing land degradation and soil erosion. Although no oil spills are visible, the mine and tailings pond clearly indicate substantial environmental risks.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Quebrada Blanca mine - Sentinel-2, December 28, 2024 (Chile)?
Answer: This Sentinel-2 image from December 28, 2024 shows the Quebrada Blanca mine in Tamarugal Province, Chile. A large tailings pond with unusually colored water signals potential chemical contamination. The image reveals widespread land degradation, with large areas of bare soil and disturbed terrain. Vegetation appears stressed or cleared, indicating habitat destruction from mining.

Question: Which environmental hazards are present in this Sentinel-2 satellite image: Fimiston Open Pit - Sentinel-2, December 25, 2024 (Australia)?
Answer: This December 25, 2024 Sentinel-2 image captures the Fimiston Open Pit, Australia's largest open-cut gold mine. A large tailings pond adjacent to the mine contains water with an abnormal color, suggesting chemical contamination. Although the image does not directly show water quality or soil contamination, the tailings pond represents a potential environmental hazard to nearby communities.

"""

# Default: use 10 examples
multi_shot_examples = multi_shot_examples_5

#KOSMOS PROMPT
common_prompt = """Identify and describe land use patterns, infrastructure, natural formations, vegetation coverage, and any visible human activities."""

#COMMON FOR BOTH MODELS
questions = ["Which types of environmental hazards and environmental degradations are present in this Sentinel-2 satellite image?"]

