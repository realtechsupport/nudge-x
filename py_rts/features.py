# io_and_features.py
import os
from typing import List, Tuple, Dict

import numpy as np
import rasterio
import imageio.v2 as imageio
from scipy import ndimage as ndi

from config import PipelineConfig


def load_multiband_and_samples(
    config: PipelineConfig,
) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    """
    Load 10-band Sentinel-2 stack and RGB scribble image, crop
    to common size, return (bands, samples_rgb, profile).

    Expected band order in GeoTIFF:
      B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
    """
    if not os.path.exists(config.multiband_path):
        raise SystemExit(f"GeoTIFF not found: {config.multiband_path}")
    if not os.path.exists(config.samples_path):
        raise SystemExit(f"Sample image not found: {config.samples_path}")

    with rasterio.open(config.multiband_path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile

    if arr.shape[0] != 10:
        raise SystemExit("Expected 10 bands (B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12).")

    (blue, green, red,
     b05, b06, b07,
     nir, b8a,
     b11, b12) = arr

    samples_rgb = imageio.imread(config.samples_path)
    if samples_rgb.ndim == 2:
        raise SystemExit("Sample image must be RGB, not grayscale.")

    H = min(blue.shape[0], samples_rgb.shape[0])
    W = min(blue.shape[1], samples_rgb.shape[1])

    bands = [b[:H, :W] for b in (blue, green, red, b05, b06, b07, nir, b8a, b11, b12)]
    samples_rgb = samples_rgb[:H, :W, :3]
    profile.update(height=H, width=W)

    return bands, samples_rgb, profile


def write_mask_tif(path: str, mask: np.ndarray, profile: dict) -> None:
    """Write a boolean mask as a single-band uint8 GeoTIFF."""
    prof = profile.copy()
    prof.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(mask.astype(np.uint8), 1)


def contrast_stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, np.float32)
    lo = np.percentile(arr[valid], p_low)
    hi = np.percentile(arr[valid], p_high)
    if hi == lo:
        lo = np.nanmin(arr); hi = np.nanmax(arr)
        if hi == lo:
            return np.zeros_like(arr, np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def hi_freq_std(img01: np.ndarray, sigma: float = 1.0, win: int = 5) -> np.ndarray:
    base = ndi.gaussian_filter(img01.astype(np.float32), sigma=sigma, mode="nearest")
    hi = np.abs(img01 - base)
    mean = ndi.uniform_filter(hi, size=win, mode="nearest")
    mean2 = ndi.uniform_filter(hi * hi, size=win, mode="nearest")
    var = np.clip(mean2 - mean * mean, 0.0, None)
    return np.sqrt(var, dtype=np.float32)


def build_feature_stack_10band(
    bands: List[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute spectral indices + texture for 10-band Sentinel-2 input.

    Returns:
      feat_stack : (H, W, D) feature cube
      aux        : dict of useful rasters (ndvi, ndwi, ndbi, B_swir, nir01, ...)
    """
    (blue, green, red,
     b05, b06, b07,
     nir, b8a,
     b11, b12) = bands

    scale = 10000.0
    blue  = np.clip(blue  / scale, 0, 1)
    green = np.clip(green / scale, 0, 1)
    red   = np.clip(red   / scale, 0, 1)
    re1   = np.clip(b05   / scale, 0, 1)
    re2   = np.clip(b06   / scale, 0, 1)
    re3   = np.clip(b07   / scale, 0, 1)
    nir   = np.clip(nir   / scale, 0, 1)
    nir_n = np.clip(b8a   / scale, 0, 1)
    swir1 = np.clip(b11   / scale, 0, 1)
    swir2 = np.clip(b12   / scale, 0, 1)

    eps = 1e-6
    ndvi   = (nir   - red)   / (nir   + red   + eps)
    ndre2  = (nir   - re2)   / (nir   + re2   + eps)
    ndwi   = (green - nir)   / (green + nir   + eps)
    ndbi11 = (swir1 - nir)   / (swir1 + nir   + eps)
    ndbi12 = (swir2 - nir)   / (swir2 + nir   + eps)
    ndmi11 = (nir   - swir1) / (nir   + swir1 + eps)
    ndmi12 = (nir   - swir2) / (nir   + swir2 + eps)
    bsi    = ((swir2 + red) - (nir + blue)) / (
             (swir2 + red) + (nir + blue) + eps)

    red01    = contrast_stretch(red)
    green01  = contrast_stretch(green)
    blue01   = contrast_stretch(blue)
    nir01    = contrast_stretch(nir)
    nir_n01  = contrast_stretch(nir_n)
    swir1_01 = contrast_stretch(swir1)
    swir2_01 = contrast_stretch(swir2)

    vis01   = (red01 + green01 + blue01) / 3.0
    B_swir  = (swir1_01 + swir2_01) / 2.0
    hf_vis  = hi_freq_std(vis01, sigma=1.0, win=5)

    feat_stack = np.stack([
        blue, green, red,
        re1, re2, re3,
        nir, nir_n,
        swir1, swir2,
        ndvi, ndre2,
        ndbi11, ndbi12,
        ndmi11, ndmi12,
        ndwi, bsi,
        vis01, B_swir,
        hf_vis,
    ], axis=-1)

    aux = dict(
        ndvi=ndvi,
        ndwi=ndwi,
        ndbi12=ndbi12,
        B_swir=B_swir,
        nir01=nir01,
    )
    return feat_stack, aux
