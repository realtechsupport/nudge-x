# classifiers.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import ndimage as ndi
from config import PipelineConfig

@dataclass
class BinaryCentroidClassifier:
    mu: np.ndarray
    sigma: np.ndarray
    c_pos: np.ndarray
    c_neg: np.ndarray
    thr: float


def train_binary_centroid_classifier(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    min_precision: float = 0.0,
) -> BinaryCentroidClassifier:
    """
    Train centroid classifier for pos vs neg.

    score = d_neg - d_pos (higher = more like positive).
    Threshold chosen to maximise F1, with optional precision floor.
    """
    X_pos = X_pos[np.all(np.isfinite(X_pos), axis=1)]
    X_neg = X_neg[np.all(np.isfinite(X_neg), axis=1)]
    if X_pos.shape[0] == 0 or X_neg.shape[0] == 0:
        raise ValueError("Need both positive and negative samples.")

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([
        np.ones(len(X_pos), dtype=np.uint8),
        np.zeros(len(X_neg), dtype=np.uint8),
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

        if f1 > best_f1_any:
            best_f1_any = f1
            best_thr_any = thr

        if precision < min_precision:
            continue

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    if best_thr is None:
        best_thr = best_thr_any
        best_f1  = best_f1_any
        print(f"  (no thr with precision >= {min_precision:.2f}, using best-F1 thr)")

    print(f"  best F1={best_f1:.3f} at thr={best_thr:.3f}")
    return (BinaryCentroidClassifier(mu=mu, sigma=sigma, c_pos=c_pos, c_neg=c_neg, thr=best_thr))


def apply_binary_centroid_classifier(X: np.ndarray, clf: BinaryCentroidClassifier) -> np.ndarray:
    Z = (X - clf.mu) / clf.sigma
    d_pos = np.sum((Z - clf.c_pos) ** 2, axis=1)
    d_neg = np.sum((Z - clf.c_neg) ** 2, axis=1)
    scores = d_neg - d_pos
    return (scores >= clf.thr)


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


def keep_components_touching_seed(mask, seed, min_area):
    """Keep only components of `mask` that intersect `seed` and are big enough."""
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


def train_scene_classifiers(
    feat_stack: np.ndarray,
    mining_sample: np.ndarray,
    urban_sample: np.ndarray,
    neg_sample: np.ndarray,
    config: PipelineConfig,
) -> Tuple[BinaryCentroidClassifier, BinaryCentroidClassifier]:
    """Train mining and urban classifiers for this scene."""
    H, W, D = feat_stack.shape
    X_all = feat_stack.reshape(-1, D)
    valid = np.all(np.isfinite(X_all), axis=1)

    mine_flat  = mining_sample.ravel()
    urban_flat = urban_sample.ravel()
    neg_flat   = neg_sample.ravel()

    X_m_pos = X_all[mine_flat & valid]
    X_u_pos = X_all[urban_flat & valid]

    neg_for_m = (urban_flat | neg_flat) & valid
    neg_for_u = (mine_flat | neg_flat) & valid

    rng = np.random.default_rng(42)

    def sample_neg(mask_flat):
        idx_all = np.where(mask_flat)[0]
        if idx_all.size > config.max_neg_train:
            return rng.choice(idx_all, size=config.max_neg_train, replace=False)
        return (idx_all)

    X_m_neg = X_all[sample_neg(neg_for_m)]
    X_u_neg = X_all[sample_neg(neg_for_u)]

    print("Training mining classifier...")
    print("  pos samples:", X_m_pos.shape[0], "neg samples:", X_m_neg.shape[0])
    mining_clf = train_binary_centroid_classifier(
        X_m_pos, X_m_neg, min_precision=config.min_precision_mining)

    print("Training urban classifier...")
    print("  pos samples:", X_u_pos.shape[0], "neg samples:", X_u_neg.shape[0])
    urban_clf = train_binary_centroid_classifier(
        X_u_pos, X_u_neg, min_precision=config.min_precision_urban)

    return (mining_clf, urban_clf)


def classify_and_postprocess(
    feat_stack: np.ndarray,
    aux: dict,
    mining_clf: BinaryCentroidClassifier,
    urban_clf: BinaryCentroidClassifier,
    mining_sample: np.ndarray,
    urban_sample: np.ndarray,
    neg_sample: np.ndarray,
    profile: dict,
    config: PipelineConfig,
):
    """
    Run classifiers on the full image and apply all spatial/spectral postprocessing.
    Returns final (mine_mask, urban_mask).
    """
    H, W, D = feat_stack.shape
    X_all = feat_stack.reshape(-1, D)
    valid_flat = np.all(np.isfinite(X_all), axis=1)
    valid = valid_flat.reshape(H, W)

    # raw classifier outputs
    mine_pred_flat  = apply_binary_centroid_classifier(X_all, mining_clf)
    urban_pred_flat = apply_binary_centroid_classifier(X_all, urban_clf)

    mine_pred  = (mine_pred_flat & valid_flat).reshape(H, W)
    urban_pred = (urban_pred_flat & valid_flat).reshape(H, W)

    ndwi   = aux["ndwi"]
    B_swir = aux["B_swir"]
    ndvi   = aux["ndvi"]
    ndbi12 = aux["ndbi12"]

    # remove obvious water
    water_like = (ndwi > 0.25) & (B_swir < 0.5)
    mine_pred[water_like]  = False
    urban_pred[water_like] = False

    # resolve overlaps
    both   = mine_pred & urban_pred
    only_m = mine_pred & ~urban_pred
    only_u = urban_pred & ~mine_pred

    mine_mask  = only_m.copy()
    urban_mask = only_u.copy()

    # where both true, use distance to positive centroids
    d_m_all = np.sum(((X_all - mining_clf.mu) / mining_clf.sigma - mining_clf.c_pos) ** 2, axis=1)
    d_u_all = np.sum(((X_all - urban_clf.mu) / urban_clf.sigma - urban_clf.c_pos) ** 2, axis=1)
    d_m_img = d_m_all.reshape(H, W)
    d_u_img = d_u_all.reshape(H, W)

    mine_mask[both & (d_m_img <  d_u_img)] = True
    urban_mask[both & (d_u_img <= d_m_img)] = True

    # always include explicit scribble samples
    mine_mask |= mining_sample
    urban_mask |= urban_sample
    urban_mask &= ~mine_mask

    # spectral gates
    if config.use_spectral_gates:
        mine_mask  &= (ndvi <= config.mining_ndvi_max) & (ndbi12 >= config.mining_ndbi_min)
        urban_mask &= (ndvi <= config.urban_ndvi_max)

    # first cleanup
    mine_mask  = cleanup_min_area(mine_mask,  config.mining_min_area, open_iter=1, close_iter=1)
    urban_mask = cleanup_min_area(urban_mask, config.urban_min_area, open_iter=1, close_iter=1)

    # negative scribbles as local veto
    if config.neg_dilate > 0:
        neg_forced = ndi.binary_dilation(neg_sample, iterations=config.neg_dilate)
    else:
        neg_forced = neg_sample
    mine_mask  &= ~neg_forced
    urban_mask &= ~neg_forced

    # optional max distance from scribbles
    if config.max_dist_px_mining is not None:
        dist_m = ndi.distance_transform_edt(~mining_sample)
        mine_mask &= (dist_m <= config.max_dist_px_mining)

    if config.max_dist_px_urban is not None:
        dist_u = ndi.distance_transform_edt(~urban_sample)
        urban_mask &= (dist_u <= config.max_dist_px_urban)

    # keep only components that actually touch scribbles
    mine_mask  = keep_components_touching_seed(mine_mask,  mining_sample, config.mining_min_area)
    urban_mask = keep_components_touching_seed(urban_mask, urban_sample,  config.urban_min_area)

    # final cleanup
    mine_mask  = cleanup_min_area(mine_mask,  config.mining_min_area, open_iter=1, close_iter=1)
    urban_mask = cleanup_min_area(urban_mask, config.urban_min_area, open_iter=1, close_iter=1)

    return (mine_mask, urban_mask)
