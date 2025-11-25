# overlay.py
import math
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import rasterio.transform as rtransform


def centroid_from_mask(mask: np.ndarray, transform) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float]
]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None, None, None, None

    cy = ys.mean()
    cx = xs.mean()

    if transform is None:
        return float(cx), float(cy), None, None

    x_map, y_map = rtransform.xy(transform, cy, cx)
    return (float(cx), float(cy), float(x_map), float(y_map))


def compute_centroid_distance(
    mine_mask: np.ndarray,
    urban_mask: np.ndarray,
    profile: dict,
):
    transform = profile.get("transform", None)
    crs = profile.get("crs", None)

    cx_m_px, cy_m_px, x_m, y_m = centroid_from_mask(mine_mask,  transform)
    cx_u_px, cy_u_px, x_u, y_u = centroid_from_mask(urban_mask, transform)

    if x_m is None or x_u is None:
        print("Centroids: one or both masks empty; no distance computed.")
        return None, None, None, None, None

    dx = x_u - x_m
    dy = y_u - y_m
    dist = math.hypot(dx, dy)

    units = "map units"
    if crs is not None and getattr(crs, "is_projected", False):
        units = "meters"

    print(f"Mining centroid (map x,y): {x_m:.2f}, {y_m:.2f}")
    print(f"Urban  centroid (map x,y): {x_u:.2f}, {y_u:.2f}")
    print(f"Distance between mining and urban centroids: {dist:.2f} {units}")

    return ((cx_m_px, cy_m_px), (cx_u_px, cy_u_px), (x_m, y_m), (x_u, y_u), dist)


def make_overlay(
    nir01: np.ndarray,
    mine_mask: np.ndarray,
    urban_mask: np.ndarray,
    mine_centroid_px=None,
    urban_centroid_px=None,
    out_path: Optional[str] = None,
):
    """Draw overlay plus optional centroid markers/line."""
    base_rgb = np.stack([nir01, nir01, nir01], axis=-1)
    class_rgb = np.zeros_like(base_rgb)
    class_rgb[mine_mask]  = (1.0, 0.0, 0.0)
    class_rgb[urban_mask] = (1.0, 1.0, 0.0)
    alpha = np.where(class_rgb.sum(axis=2) > 0, 0.55, 0.0)[..., None]
    overlay = base_rgb * (1 - alpha) + class_rgb * alpha

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)

    if mine_centroid_px and urban_centroid_px:
        cx_m, cy_m = mine_centroid_px
        cx_u, cy_u = urban_centroid_px
        plt.plot([cx_m, cx_u], [cy_m, cy_u],
                 linestyle="--", linewidth=2, color="white")
        plt.scatter([cx_m], [cy_m], marker="x", s=80, c="cyan",   label="mine centroid")
        plt.scatter([cx_u], [cy_u], marker="x", s=80, c="magenta", label="urban centroid")
        plt.legend(loc="upper right", fontsize=8)

    plt.title("Mining (red) and urban (yellow)")
    plt.axis("off")
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.show()
