# -*- coding: utf-8 -*-
"""
urban_mining_refactored_exact.py

Behavior-preserving refactor of the original Colab-exported urban_mining.py.
Goal: reorganize the operational ~300-line pipeline into clear sub-functions
WITHOUT changing any algorithmic behavior, thresholds, or helper implementations.

IMPORTANT:
- This file continues to import and rely on `urban_mining_helper` for core helper
  functions (contrast_stretch, hi_freq_std, extract_masks_from_colors,
  train_binary_centroid_classifier, apply_binary_centroid_classifier,
  cleanup_min_area, keep_components_touching_seed, centroid_from_mask,
  component_centroids, etc.). Those implementations are NOT duplicated here,
  to avoid drifting behavior.

Author: (refactor) ChatGPT
"""

import os, math
import numpy as np
import rasterio
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# Use the original helper implementations (behavior-critical)
from urban_mining_helper import *  # noqa: F401,F403


# ---------------------------------------------------------------------
# USER CONFIGURATION (keep as in your notebook / script)
# ---------------------------------------------------------------------

# Example defaults (edit as needed)
# MULTIBAND_PATH = ...
# SAMPLES_PATH = ...
# OUT_PREFIX = ...

# MINING_COLORS = [(255,0,0)]
# URBAN_COLORS  = [(255,255,0)]
# NEGATIVE_COLORS = [(0,0,255)]
# COLOR_TOL = 25
# MAX_NEG_TRAIN = 50000
# MINING_MIN_AREA = 2000
# URBAN_MIN_AREA = 800
# MIN_PRECISION_MINING = 0.85
# MIN_PRECISION_URBAN = 0.80
# NEG_DILATE = 3
# MAX_DIST_PX_MINING = 300
# MAX_DIST_PX_URBAN = 300
# USE_SPECTRAL_GATES = True
# MINING_NDVI_MAX = 0.45
# MINING_NDBI_MIN = 0.00
# URBAN_NDVI_MAX = 0.50


# -------------------------------
# Small utilities (non-behavioral)
# -------------------------------

def to_u8_colors(color_list):
    """Lifted from the original nested helper inside main()."""
    out = []
    for c in color_list:
        c = tuple(c)
        if max(c) <= 1.0:
            out.append(tuple(int(round(x * 255.0)) for x in c))
        else:
            out.append(tuple(int(round(x)) for x in c))
    return out


# -------------------------------
# Pipeline functions (behavior-preserving)
# -------------------------------

def load_multiband_stack(multiband_path):
    if not os.path.exists(multiband_path):
        raise SystemExit(f"GeoTIFF not found: {multiband_path}")

    with rasterio.open(multiband_path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile

    if arr.shape[0] != 10:
        raise SystemExit("Expected 10 bands (B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12).")

    (blue, green, red,
     b05, b06, b07,
     nir, b8a,
     b11, b12) = arr

    return (blue, green, red, b05, b06, b07, nir, b8a, b11, b12), profile


def load_and_align_samples(samples_path, bands, profile):
    """
    EXACT behavior from the provided baseline:
    - Load RGB/RGBA scribble image
    - Enforce uint8 0..255
    - Prefer resize with PIL NEAREST to match GeoTIFF
    - Fallback: top-left crop both raster bands and scribbles to common min(H,W)
    """
    if not os.path.exists(samples_path):
        raise SystemExit(f"Sample image not found: {samples_path}")

    (blue, green, red, b05, b06, b07, nir, b8a, b11, b12) = bands
    H_tif, W_tif = blue.shape

    samples_rgb = imageio.imread(samples_path)

    if samples_rgb.ndim == 2:
        raise SystemExit("Sample image must be RGB/RGBA, not grayscale.")
    if samples_rgb.shape[2] >= 3:
        samples_rgb = samples_rgb[:, :, :3]
    else:
        raise SystemExit("Sample image must have at least 3 channels (RGB).")

    if samples_rgb.dtype != np.uint8:
        if np.issubdtype(samples_rgb.dtype, np.floating) and float(samples_rgb.max()) <= 1.0:
            samples_rgb = (samples_rgb * 255.0).round().astype(np.uint8)
        else:
            samples_rgb = np.clip(samples_rgb, 0, 255).astype(np.uint8)

    H_s, W_s = samples_rgb.shape[:2]

    resized_ok = False
    if (H_s, W_s) != (H_tif, W_tif):
        try:
            from PIL import Image
            samples_rgb = np.array(
                Image.fromarray(samples_rgb).resize((W_tif, H_tif), resample=Image.Resampling.NEAREST),
                dtype=np.uint8
            )
            print(f"Resized sample image from {(H_s, W_s)} -> {(H_tif, W_tif)} using NEAREST.")
            resized_ok = True
        except Exception as e:
            print(f"WARNING: Could not resize with PIL ({e}). Falling back to top-left crop.")
            resized_ok = False

    if not resized_ok:
        H = min(H_tif, H_s)
        W = min(W_tif, W_s)
        blue, green, red, b05, b06, b07, nir, b8a, b11, b12 = [
            b[:H, :W] for b in (blue, green, red, b05, b06, b07, nir, b8a, b11, b12)
        ]
        samples_rgb = samples_rgb[:H, :W, :3]
        profile.update(height=H, width=W)
    else:
        profile.update(height=H_tif, width=W_tif)

    # Ensure H,W reflect the (possibly cropped) band stack shape
    H, W = blue.shape
    bands_aligned = (blue, green, red, b05, b06, b07, nir, b8a, b11, b12)
    return samples_rgb, bands_aligned, profile, (H, W)


def compute_features(bands_aligned):
    (blue, green, red, b05, b06, b07, nir, b8a, b11, b12) = bands_aligned

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
    ndbi12 = (swir2 - nir)   / (swir2 + nir   + eps)
    ndbi11 = (swir1 - nir)   / (swir1 + nir   + eps)
    ndmi11 = (nir   - swir1) / (nir   + swir1 + eps)
    ndmi12 = (nir   - swir2) / (nir   + swir2 + eps)
    bsi    = ((swir2 + red) - (nir + blue)) / ((swir2 + red) + (nir + blue) + eps)

    red01    = contrast_stretch(red)
    green01  = contrast_stretch(green)
    blue01   = contrast_stretch(blue)
    nir01    = contrast_stretch(nir)
    nir_n01  = contrast_stretch(nir_n)
    re201    = contrast_stretch(re2)
    swir1_01 = contrast_stretch(swir1)
    swir2_01 = contrast_stretch(swir2)

    vis01  = (red01 + green01 + blue01) / 3.0
    B_nir  = (nir01 + nir_n01) / 2.0
    B_swir = (swir1_01 + swir2_01) / 2.0

    hf_vis = hi_freq_std(vis01, sigma=1.0, win=5)

    feat_stack = np.stack([
        blue, green, red,
        re1, re2, re3,
        nir, nir_n,
        swir1, swir2,
        ndvi, ndre2,
        ndbi11, ndbi12,
        ndmi11, ndmi12,
        ndwi, bsi,
        vis01, B_nir, B_swir,
        hf_vis,
    ], axis=-1)

    H, W, D = feat_stack.shape
    X_all = feat_stack.reshape(-1, D)
    valid = np.all(np.isfinite(X_all), axis=1)

    aux = {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndbi12": ndbi12,
        "B_swir": B_swir,
        "nir": nir,
    }
    return feat_stack, X_all, valid, aux


def extract_samples(samples_rgb, H, W,
                    mining_colors, urban_colors, negative_colors,
                    color_tol, color_tol_u8_min):
    # EXACT diagnostics + extraction logic from baseline
    if samples_rgb.shape[0] != H or samples_rgb.shape[1] != W:
        samples_rgb = samples_rgb[:H, :W, :3]

    if samples_rgb.dtype != np.uint8:
        samples_rgb = np.clip(samples_rgb, 0, 255).astype(np.uint8)

    print("samples_rgb dtype/min/max:", samples_rgb.dtype, int(samples_rgb.min()), int(samples_rgb.max()))

    flat = samples_rgb.reshape(-1, 3)
    bright = flat[(flat.max(axis=1) > 240) & (flat.min(axis=1) < 60)]
    if bright.size > 0:
        vals, counts = np.unique(bright, axis=0, return_counts=True)
        order = np.argsort(-counts)
        top = list(zip(map(tuple, vals[order[:10]]), counts[order[:10]]))
        print("Top bright colors (RGB,count):", top)
    else:
        print("No bright label-like pixels found (check annotation image / resizing).")

    mining_u8   = to_u8_colors(mining_colors)
    urban_u8    = to_u8_colors(urban_colors)
    negative_u8 = to_u8_colors(negative_colors)

    color_tol_u8 = color_tol
    if isinstance(color_tol, float) and color_tol <= 1.0:
        color_tol_u8 = int(round(color_tol * 255.0))
    color_tol_u8 = max(int(color_tol_u8), int(color_tol_u8_min))

    mining_sample, urban_sample, neg_sample = extract_masks_from_colors(
        samples_rgb,
        [mining_u8, urban_u8, negative_u8],
        color_tol_u8
    )

    print("Sample pixels – mining:", int(mining_sample.sum()),
          "urban:", int(urban_sample.sum()),
          "negative:", int(neg_sample.sum()))

    if mining_sample.sum() < 50 or urban_sample.sum() < 50 or neg_sample.sum() < 50:
        raise SystemExit("Need at least ~50 pixels for each of mining, urban, and negative samples.")

    return samples_rgb, mining_sample, urban_sample, neg_sample


def train_classifiers(X_all, valid, mining_sample, urban_sample, neg_sample,
                      max_neg_train, min_precision_mining, min_precision_urban):
    mine_flat  = mining_sample.ravel()
    urban_flat = urban_sample.ravel()
    neg_flat   = neg_sample.ravel()

    rng = np.random.default_rng(42)

    # Mining classifier: pos=mining, neg=urban+negative
    X_m_pos = X_all[mine_flat & valid]
    neg_for_m = (urban_flat | neg_flat) & valid
    neg_idx_all = np.where(neg_for_m)[0]
    if neg_idx_all.size > max_neg_train:
        neg_idx = rng.choice(neg_idx_all, size=max_neg_train, replace=False)
    else:
        neg_idx = neg_idx_all
    X_m_neg = X_all[neg_idx]

    print("Training mining classifier...")
    print("  pos samples:", X_m_pos.shape[0], "neg samples:", X_m_neg.shape[0])
    mu_m, sig_m, c_m_pos, c_m_neg, thr_m = train_binary_centroid_classifier(
        X_m_pos, X_m_neg, min_precision=min_precision_mining
    )

    # Urban classifier: pos=urban, neg=mining+negative
    X_u_pos = X_all[urban_flat & valid]
    neg_for_u = (mine_flat | neg_flat) & valid
    neg_u_idx_all = np.where(neg_for_u)[0]
    if neg_u_idx_all.size > max_neg_train:
        neg_u_idx = rng.choice(neg_u_idx_all, size=max_neg_train, replace=False)
    else:
        neg_u_idx = neg_u_idx_all
    X_u_neg = X_all[neg_u_idx]

    print("Training urban classifier...")
    print("  pos samples:", X_u_pos.shape[0], "neg samples:", X_u_neg.shape[0])
    mu_u, sig_u, c_u_pos, c_u_neg, thr_u = train_binary_centroid_classifier(
        X_u_pos, X_u_neg, min_precision=min_precision_urban
    )

    return (mu_m, sig_m, c_m_pos, c_m_neg, thr_m,
            mu_u, sig_u, c_u_pos, c_u_neg, thr_u)


def apply_and_postprocess(X_all, valid, H, W,
                          mining_sample, urban_sample, neg_sample,
                          aux,
                          model_tuple,
                          mine_water_keep_dist_px,
                          mining_gate_relax_dist_px,
                          use_spectral_gates,
                          mining_ndvi_max, mining_ndbi_min, urban_ndvi_max,
                          mining_min_area, urban_min_area,
                          neg_dilate,
                          max_dist_px_mining, max_dist_px_urban):
    (mu_m, sig_m, c_m_pos, c_m_neg, thr_m,
     mu_u, sig_u, c_u_pos, c_u_neg, thr_u) = model_tuple

    print("Applying classifiers to full image...")
    mine_pred_flat = apply_binary_centroid_classifier(
        X_all, mu_m, sig_m, c_m_pos, c_m_neg, thr_m
    )
    urban_pred_flat = apply_binary_centroid_classifier(
        X_all, mu_u, sig_u, c_u_pos, c_u_neg, thr_u
    )

    mine_pred  = mine_pred_flat.reshape(H, W) & valid.reshape(H, W)
    urban_pred = urban_pred_flat.reshape(H, W) & valid.reshape(H, W)

    # Fix A: water-like veto, keep water-like mining near mine seeds
    water_like = (aux["ndwi"] > 0.25) & (aux["B_swir"] < 0.5)
    dist_to_mine_seed = ndi.distance_transform_edt(~mining_sample)
    mine_water_keep = dist_to_mine_seed <= mine_water_keep_dist_px
    mine_pred[water_like & ~mine_water_keep] = False
    urban_pred[water_like] = False

    # resolve overlaps
    both = mine_pred & urban_pred
    only_m = mine_pred & ~urban_pred
    only_u = urban_pred & ~mine_pred

    mine_mask  = only_m.copy()
    urban_mask = only_u.copy()

    d_m_all = np.sum(((X_all - mu_m) / sig_m - c_m_pos) ** 2, axis=1)
    d_u_all = np.sum(((X_all - mu_u) / sig_u - c_u_pos) ** 2, axis=1)
    d_m_img = d_m_all.reshape(H, W)
    d_u_img = d_u_all.reshape(H, W)
    mine_mask[both & (d_m_img <  d_u_img)] = True
    urban_mask[both & (d_u_img <= d_m_img)] = True

    # always include explicit samples
    mine_mask |= mining_sample
    urban_mask |= urban_sample
    urban_mask &= ~mine_mask

    # Fix C: relax mining spectral gates near mining seeds
    if use_spectral_gates:
        dist_m_gate = ndi.distance_transform_edt(~mining_sample)
        near_mine = dist_m_gate <= mining_gate_relax_dist_px

        strict_mining = (aux["ndvi"] <= mining_ndvi_max) & (aux["ndbi12"] >= mining_ndbi_min)
        relaxed_mining = (aux["ndvi"] <= mining_ndvi_max)

        mine_mask &= (strict_mining | (near_mine & relaxed_mining))
        urban_mask &= (aux["ndvi"] <= urban_ndvi_max)

    # morphology cleanup (first pass)
    mine_mask  = cleanup_min_area(mine_mask,  mining_min_area, open_iter=1, close_iter=1)
    urban_mask = cleanup_min_area(urban_mask, urban_min_area,  open_iter=1, close_iter=1)

    # negative veto
    if neg_dilate > 0:
        neg_forced = ndi.binary_dilation(neg_sample, iterations=neg_dilate)
    else:
        neg_forced = neg_sample
    mine_mask  &= ~neg_forced
    urban_mask &= ~neg_forced

    # max-distance constraints
    if max_dist_px_mining is not None:
        dist_m = ndi.distance_transform_edt(~mining_sample)
        mine_mask &= (dist_m <= max_dist_px_mining)

    if max_dist_px_urban is not None:
        dist_u = ndi.distance_transform_edt(~urban_sample)
        urban_mask &= (dist_u <= max_dist_px_urban)

    # restrict to components touching seeds
    mine_mask  = keep_components_touching_seed(mine_mask,  mining_sample, mining_min_area)
    urban_mask = keep_components_touching_seed(urban_mask, urban_sample,  urban_min_area)

    # final cleanup
    mine_mask  = cleanup_min_area(mine_mask,  mining_min_area, open_iter=1, close_iter=1)
    urban_mask = cleanup_min_area(urban_mask, urban_min_area,  open_iter=1, close_iter=1)

    # Fix B: fill internal holes in mining components
    mine_mask = ndi.binary_fill_holes(mine_mask)
    urban_mask &= ~mine_mask

    return mine_mask, urban_mask


def compute_centroids_and_report(mine_mask, urban_mask, profile):
    transform = profile.get("transform", None)
    crs = profile.get("crs", None)

    cx_m_px, cy_m_px, x_m, y_m = centroid_from_mask(mine_mask, transform)
    cx_u_px, cy_u_px, x_u, y_u = centroid_from_mask(urban_mask, transform)

    if x_m is None or x_u is None:
        print("Centroids: cannot compute distance – one of the masks is empty.")
    else:
        dx = x_u - x_m
        dy = y_u - y_m
        dist_global = math.hypot(dx, dy)

        units = "map units"
        if crs is not None and crs.is_projected:
            units = "meters"

        print(f"Mining centroid (map x,y): {x_m:.2f}, {y_m:.2f}")
        print(f"Urban  centroid (map x,y): {x_u:.2f}, {y_u:.2f}")
        print(f"Distance between mining and urban centroids: {dist_global:.2f} {units}")

    mine_centers  = component_centroids(mine_mask,  transform)
    urban_centers = component_centroids(urban_mask, transform)

    print("\nComponent centroids:")
    print(f"  Mining components: {len(mine_centers)}")
    print(f"  Urban  components: {len(urban_centers)}")

    if mine_centers and urban_centers and (mine_centers[0][2] is not None):
        for i, (cx_m, cy_m, x_mi, y_mi) in enumerate(mine_centers):
            dists = [
                np.sqrt((x_mi - x_ui) ** 2 + (y_mi - y_ui) ** 2)
                for (_, _, x_ui, y_ui) in urban_centers
            ]
            d_min = float(np.min(dists)) if dists else float("nan")
            print(f"  Mine #{i+1} at ({x_mi:.1f},{y_mi:.1f}) – "
                  f"nearest urban centroid: {d_min:.1f} map units away")
    else:
        print("Not enough centroids to compute per-component distances.")

    return cx_m_px, cy_m_px, cx_u_px, cy_u_px


def overlay_and_save(mine_mask, urban_mask, nir01, out_prefix, cx_m_px, cy_m_px, cx_u_px, cy_u_px):
    base = contrast_stretch(nir01)  # NIR as grayscale background
    base_rgb = np.stack([base, base, base], axis=-1)
    class_rgb = np.zeros_like(base_rgb)
    class_rgb[mine_mask]  = (1.0, 0.0, 0.0)
    class_rgb[urban_mask] = (1.0, 1.0, 0.0)
    alpha = np.where(class_rgb.sum(axis=2) > 0, 0.55, 0.0)[..., None]
    overlay = base_rgb * (1 - alpha) + class_rgb * alpha

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title("Mining (red) and urban (yellow)")
    plt.axis("off")

    if cx_m_px is not None and cy_m_px is not None and cx_u_px is not None and cy_u_px is not None:
        plt.scatter([cx_m_px], [cy_m_px], marker="x", s=80, c="cyan",    label="mine centroid")
        plt.scatter([cx_u_px], [cy_u_px], marker="x", s=80, c="magenta", label="urban centroid")

    plt.savefig(f"{out_prefix}_overlay.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.show()

    print("Saved:")
    print("  ", f"{out_prefix}_overlay.png")


# ---------------------------------------------------------------------
# MAIN (refactored, same behavior)
# ---------------------------------------------------------------------

def main():
    # Tunable parameters (local) — unchanged
    MINE_WATER_KEEP_DIST_PX = 30
    COLOR_TOL_U8_MIN = 3
    MINING_GATE_RELAX_DIST_PX = 70

    if not os.path.exists(MULTIBAND_PATH):
        raise SystemExit(f"GeoTIFF not found: {MULTIBAND_PATH}")
    if not os.path.exists(SAMPLES_PATH):
        raise SystemExit(f"Sample image not found: {SAMPLES_PATH}")

    bands, profile = load_multiband_stack(MULTIBAND_PATH)
    samples_rgb, bands_aligned, profile, (H, W) = load_and_align_samples(SAMPLES_PATH, bands, profile)

    feat_stack, X_all, valid, aux = compute_features(bands_aligned)
    H2, W2, _ = feat_stack.shape
    # keep exact behavior: enforce samples dims to feature dims before extraction
    samples_rgb, mining_sample, urban_sample, neg_sample = extract_samples(
        samples_rgb, H2, W2,
        MINING_COLORS, URBAN_COLORS, NEGATIVE_COLORS,
        COLOR_TOL, COLOR_TOL_U8_MIN
    )

    model_tuple = train_classifiers(
        X_all, valid, mining_sample, urban_sample, neg_sample,
        MAX_NEG_TRAIN, MIN_PRECISION_MINING, MIN_PRECISION_URBAN
    )

    mine_mask, urban_mask = apply_and_postprocess(
        X_all, valid, H2, W2,
        mining_sample, urban_sample, neg_sample,
        aux,
        model_tuple,
        MINE_WATER_KEEP_DIST_PX,
        MINING_GATE_RELAX_DIST_PX,
        USE_SPECTRAL_GATES,
        MINING_NDVI_MAX, MINING_NDBI_MIN, URBAN_NDVI_MAX,
        MINING_MIN_AREA, URBAN_MIN_AREA,
        NEG_DILATE,
        MAX_DIST_PX_MINING, MAX_DIST_PX_URBAN
    )

    print("Final mining pixels:", int(mine_mask.sum()))
    print("Final urban pixels :", int(urban_mask.sum()))
    print("Extra mining pixels outside samples:", int((mine_mask & ~mining_sample).sum()))
    print("Extra urban pixels outside samples :", int((urban_mask & ~urban_sample).sum()))

    cx_m_px, cy_m_px, cx_u_px, cy_u_px = compute_centroids_and_report(mine_mask, urban_mask, profile)

    # use nir already scaled in aux for overlay background (matches baseline's `nir`)
    overlay_and_save(mine_mask, urban_mask, aux["nir"], OUT_PREFIX, cx_m_px, cy_m_px, cx_u_px, cy_u_px)


if __name__ == "__main__":
    main()
