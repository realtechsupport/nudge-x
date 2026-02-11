# samples.py
from typing import List, Tuple
import numpy as np
from scipy import ndimage as ndi


def extract_masks_from_colors(
    samples_rgb: np.ndarray,
    color_lists: List[List[Tuple[int, int, int]]],
    tol: int,
):
    """
    samples_rgb : (H, W, 3) uint8
    color_lists : e.g. [MINING_COLORS, URBAN_COLORS, NEGATIVE_COLORS]
    tol         : L1 distance in RGB
    returns     : list of boolean masks, one per color_list
    """
    rgb = samples_rgb[..., :3].astype(np.int16)
    H, W, _ = rgb.shape
    masks = []

    for color_list in color_lists:
        outline = np.zeros((H, W), bool)
        for (r, g, b) in color_list:
            target = np.array([r, g, b], np.int16).reshape(1, 1, 3)
            diff = np.abs(rgb - target).sum(axis=2)
            outline |= (diff <= tol)
        if outline.any():
            mask = ndi.binary_fill_holes(outline)
        else:
            mask = np.zeros_like(outline)
        masks.append(mask)

    return masks  # [mining_sample, urban_sample, negative_sample]
