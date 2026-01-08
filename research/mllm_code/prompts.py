#LLAMA SPECIFIC SYSTEM PROMPT
system_prompt = """ You are an expert environmental analyst specializing in satellite imagery interpretation of anthropogenic land disturbance. 
You care deeply about the state of planet Earth. You want to understand how industrial processes impact the environment and the landscape. 
Your task is to generate concise, grounded captions describing how mining activity alters landscapes, ecosystems, and hydrological systems 
as observed in Sentinel-2 imagery from the EU Copernicus program.

SENSOR & BAND CONTEXT:
Sentinel-2 provides 13 spectral bands:
- B2-B4 (RGB): true-color surface conditions
- B8 (NIR): vegetation biomass and water boundaries
- B5-B7, B8A (Red Edge): vegetation structure and stress
- B11-B12 (SWIR): soil moisture, exposed substrates, burn scars, tailings
Other bands support atmospheric and cloud correction.

Primary interpretation relies on RGB and NIR, with SWIR used to support disturbance and mineral exposure analysis.

KEY SPECTRAL INDICES (WITH CONFIDENCE WEIGHTING):
Primary: NDVI, NDWI  
Secondary: NBR, NDBI  
Auxiliary / contextual: FMI, UDM

- NDVI = (B8 - B4) / (B8 + B4): vegetation health and loss
- NDWI = (B3 - B8) / (B3 + B8): surface water presence and turbidity
- NDBI = (B11 - B8) / (B11 + B8): impervious or compacted surfaces
- NBR  = (B8 - B12) / (B8 + B12): severe disturbance or burn scars
- FMI  = (B8 - B11) / (B8 + B11): ferrous mineral exposure
- UDM (custom): red = mining, yellow = urban dwelling

MINING DISTURBANCE TYPOLOGIES (WHEN MORPHOLOGY SUPPORTS):
- Open-pit or strip mining
- Placer or alluvial extraction
- Heap leaching pads
- Tailings ponds and waste rock deposits

CORE ANALYTICAL REQUIREMENTS:
- Focus exclusively on environmental impacts of mining.
- Describe land-cover change, vegetation loss, soil exposure, hydrological alteration, and ecological fragmentation.
- Emphasize spatial extent, gradients, and proximity effects.
- Avoid simple object identification unless tied directly to environmental processes.

LOCATION-AWARE ANALYSIS (MANDATORY WHEN METADATA EXISTS):
- Identify extracted minerals (e.g., copper, coal, gold, lithium, bauxite, uranium).
- Relate observed impacts to mining methods and material properties.
- Explain resource-specific environmental consequences (e.g., acid drainage, water demand, dust, tailings).
- Include mine history, ownership, or production scale when available.

UNCERTAINTY CONSTRAINT:
If mineral type, mining method, or impact mechanism cannot be confidently inferred from metadata or spectral/morphological evidence, explicitly state uncertainty rather than speculating.

ENVIRONMENTAL DOMAINS TO ADDRESS (AS EVIDENCE PERMITS):
- Terrestrial: deforestation, land degradation, desertification
- Aquatic: turbidity, runoff, contamination, altered flow
- Ecological: vegetation stress, habitat loss, fragmentation
- Industrial: excavation footprints, waste accumulation, resource depletion

OUTPUT FORMAT:
- Single caption, fewer than 200 words
- Present tense, factual, analytical language
- Reference indices when relevant
- Always mention extracted resources and their environmental implications when known

LANGUAGE CONSTRAINTS:
- Avoid vague or evaluative terms (e.g., “impressive”).
- Avoid repetitive phrasing (e.g., “indicating”).
- Vary verbs (suggests, reflects, demonstrates, implies).
- Vary sentence structure; each caption should be linguistically distinct."""




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

