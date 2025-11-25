# config.py
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os


@dataclass
class PipelineConfig:
    multiband_path: str
    samples_path: str
    out_prefix: str

    mining_colors: List[Tuple[int, int, int]]
    urban_colors: List[Tuple[int, int, int]]
    negative_colors: List[Tuple[int, int, int]]

    color_tol: int
    max_neg_train: int
    mining_min_area: int
    urban_min_area: int

    min_precision_mining: float
    min_precision_urban: float

    neg_dilate: int
    max_dist_px_mining: Optional[int]
    max_dist_px_urban: Optional[int]

    use_spectral_gates: bool
    mining_ndvi_max: float
    mining_ndbi_min: float
    urban_ndvi_max: float


def make_default_config(multiband_path: str, samples_path: str) -> PipelineConfig:
    """
    Create a default config for a scene.

    Adjust paths and defaults as needed.
    """
    out_prefix = os.path.splitext(multiband_path)[0]

    return PipelineConfig(
        multiband_path=multiband_path,
        samples_path=samples_path,
        out_prefix=out_prefix,

        mining_colors=[(255, 0, 0)],          # red scribbles
        urban_colors=[(255, 255, 0)],         # yellow scribbles
        negative_colors=[(0, 0, 255)],        # blue scribbles

        color_tol=25,
        max_neg_train=50000,
        mining_min_area=2000,
        urban_min_area=500,

        min_precision_mining=0.85,
        min_precision_urban=0.80,

        neg_dilate=3,
        max_dist_px_mining=None,              # e.g. 250 to limit range
        max_dist_px_urban=None,

        use_spectral_gates=True,
        mining_ndvi_max=0.45,
        mining_ndbi_min=0.0,
        urban_ndvi_max=0.70,
    )
