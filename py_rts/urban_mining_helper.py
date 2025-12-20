# urban_mining_helper.py
# Helper functions for Sentinel-2 image based detction of urban and mining in proximity
# -*- coding: utf-8 -*-
# Nov 2025
#
"""
Urban / mining classifier using positive + negative scribble samples,
for a 10-band Sentinel-2 GeoTIFF (in this EXACT band order):

    band 1: B02  (blue,  10 m)
    band 2: B03  (green, 10 m)
    band 3: B04  (red,   10 m)
    band 4: B05  (red-edge 1, 20 m resampled to 10 m)
    band 5: B06  (red-edge 2, 20 m -> 10 m)
    band 6: B07  (red-edge 3, 20 m -> 10 m)
    band 7: B08  (NIR broad, 10 m)
    band 8: B8A  (NIR narrow,20 m -> 10 m)
    band 9: B11  (SWIR1, 20 m -> 10 m)
    band10: B12  (SWIR2, 20 m -> 10 m)

Inputs
------
1) MULTIBAND_PATH : path to the 10-band GeoTIFF above
2) SAMPLES_PATH   : RGB image, same HxW as GeoTIFF, with scribbles:
       - mining : one or more colors in MINING_COLORS
       - urban  : one or more colors in URBAN_COLORS
       - neg    : one or more colors in NEGATIVE_COLORS
         (neg = "neither mining nor urban")

Outputs (OUT_PREFIX = MULTIBAND_PATH basename)
----------------------------------------------
- OUT_PREFIX_mining_mask.(tif|png)
- OUT_PREFIX_urban_mask.(tif|png)
- OUT_PREFIX_overlay.png   (NIR background, mining=red, urban=yellow)
"""

import os, math
import numpy as np
from scipy import ndimage as ndi
import rasterio.transform as rtransform
from scipy import ndimage as ndi

# -------------------------------------------------------------------------------------------------
def contrast_stretch(arr, p_low=2, p_high=98):
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, np.float32)
    lo = np.percentile(arr[valid], p_low)
    hi = np.percentile(arr[valid], p_high)
    if hi == lo:
        lo = np.nanmin(arr); hi = np.nanmax(arr)
        if hi == lo:
            return (np.zeros_like(arr, np.float32))
    out = (arr - lo) / (hi - lo)
    return (np.clip(out, 0, 1).astype(np.float32))


def hi_freq_std(img01, sigma=1.0, win=5):
    base = ndi.gaussian_filter(img01.astype(np.float32), sigma=sigma, mode="nearest")
    hi = np.abs(img01 - base)
    mean = ndi.uniform_filter(hi, size=win, mode="nearest")
    mean2 = ndi.uniform_filter(hi * hi, size=win, mode="nearest")
    var = np.clip(mean2 - mean * mean, 0.0, None)
    return (np.sqrt(var, dtype=np.float32))


def cleanup_min_area(mask, min_area, open_iter=1, close_iter=1):
    struct = np.ones((3, 3), bool)
    m = mask.copy()
    if open_iter > 0:
        m = ndi.binary_opening(m, structure=struct, iterations=open_iter)
    if close_iter > 0:
        m = ndi.binary_closing(m, structure=struct, iterations=close_iter)
    lbl, nlab = ndi.label(m, structure=struct)
    if nlab == 0:
        return m
    areas = ndi.sum(np.ones_like(m, np.float32), lbl, index=range(1, nlab + 1))
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_area]
    if not keep:
        return np.zeros_like(mask, bool)
    return (np.isin(lbl, keep))


def write_mask_tif(path, mask, ref_profile):
    prof = ref_profile.copy()
    prof.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(mask.astype(np.uint8), 1)


def extract_masks_from_colors(samples_rgb, color_lists, tol):
    """
    samples_rgb : (H, W, 3) uint8
    color_lists : list of list-of-RGB, e.g. [MINING_COLORS, URBAN_COLORS, NEGATIVE_COLORS]
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

    return (masks)  # [mining_sample, urban_sample, negative_sample]


def keep_components_touching_seed(mask, seed, min_area):
    """Keep only components of `mask` that intersect `seed` and are large enough."""
    struct = np.ones((3, 3), bool)
    lbl, nlab = ndi.label(mask, structure=struct)
    if nlab == 0:
        return np.zeros_like(mask, bool)
    seed_labels = np.unique(lbl[seed])
    keep_mask = np.zeros_like(mask, bool)
    for lab in seed_labels:
        if lab == 0:
            continue
        comp = (lbl == lab)
        if comp.sum() >= min_area:
            keep_mask |= comp
    return (keep_mask)


def train_binary_centroid_classifier(X_pos, X_neg, min_precision=0.0):
    """
    Train centroid classifier for pos vs neg.

    Returns (mu, sigma, c_pos, c_neg, best_thr)

    score = d_neg - d_pos  (higher = more like positive).

    Threshold chosen to maximise F1, but only among thresholds with
    precision >= min_precision. Fallback to best-F1 if none meet the floor.
    """
    X_pos = X_pos[np.all(np.isfinite(X_pos), axis=1)]
    X_neg = X_neg[np.all(np.isfinite(X_neg), axis=1)]
    if X_pos.shape[0] == 0 or X_neg.shape[0] == 0:
        raise ValueError("Need both positive and negative samples.")

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([
        np.ones(len(X_pos), dtype=np.uint8),
        np.zeros(len(X_neg), dtype=np.uint8)
    ])

    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0) + 1e-6
    Z = (X - mu) / sigma

    Z_pos = Z[y == 1]
    Z_neg = Z[y == 0]
    c_pos = np.nanmean(Z_pos, axis=0)
    c_neg = np.nanmean(Z_neg, axis=0)

    d_pos = np.sum((Z - c_pos) ** 2, axis=1)
    d_neg = np.sum((Z - c_neg) ** 2, axis=1)
    scores = d_neg - d_pos  # higher = more like positive

    qs = np.linspace(0.1, 0.9, 25)
    thr_candidates = np.quantile(scores, qs)

    best_thr_any = np.median(scores)
    best_f1_any  = -1.0
    best_thr = None
    best_f1  = -1.0

    for thr in thr_candidates:
        y_pred = (scores >= thr).astype(np.uint8)
        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        # global best-F1 (fallback)
        if f1 > best_f1_any:
            best_f1_any = f1
            best_thr_any = thr

        # precision floor
        if precision < min_precision:
            continue

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    if best_thr is None:
        best_thr = best_thr_any
        best_f1  = best_f1_any

    return (mu, sigma, c_pos, c_neg, best_thr)


def apply_binary_centroid_classifier(X, mu, sigma, c_pos, c_neg, thr):
    Z = (X - mu) / sigma
    d_pos = np.sum((Z - c_pos) ** 2, axis=1)
    d_neg = np.sum((Z - c_neg) ** 2, axis=1)
    scores = d_neg - d_pos
    return (scores >= thr)


def centroid_from_mask(mask, transform):
    """
    Compute centroid of a boolean mask.
    Returns:
        (cx_px, cy_px, x_map, y_map)
      - cx_px, cy_px : centroid in pixel coordinates (col, row)
      - x_map, y_map : centroid in map coordinates (x, y) via `transform` or (None, None) if transform is None.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None, None, None, None

    cy = ys.mean()
    cx = xs.mean()

    if transform is None:
        return cx, cy, None, None

    # centre of pixel (cx, cy)
    x_map, y_map = rtransform.xy(transform, cy, cx)
    return (float(cx), float(cy), float(x_map), float(y_map))


def component_centroids(mask, transform=None):
    """
    Centroids for each connected component in `mask`.
    Returns a list of tuples:[(cx_px, cy_px, cx_map, cy_map), ...]
    """
    struct = np.ones((3, 3), bool)
    labels, nlab = ndi.label(mask, structure=struct)
    centers = []

    for lab in range(1, nlab + 1):
        ys, xs = np.nonzero(labels == lab)
        if ys.size == 0:
            continue
        cy = ys.mean()
        cx = xs.mean()
        if transform is not None:
            x_map, y_map = rtransform.xy(transform, cy, cx)
        else:
            x_map = y_map = None
        centers.append((cx, cy, x_map, y_map))

    return (centers)
 # -------------------------------------------------------------------------------------------------  
