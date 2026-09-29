"""Gaussian-kernel density maps whose integral equals the object count."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def density_map(
    points: np.ndarray, shape: tuple[int, int], sigma: float = 15.0, scale: int = 1
) -> np.ndarray:
    """Place a unit impulse per point and blur with a Gaussian.

    ``points`` is (N, 2) in (x, y) pixels, ``shape`` is (H, W). Reflective boundary handling
    conserves mass, so ``density_map(...).sum() == N`` up to floating point, even for points
    near the border. Points outside the image are clipped to the nearest pixel. With
    ``scale > 1`` the map is computed on a grid ``scale`` times coarser (same integral, faster).
    """
    h, w = shape[0] // scale, shape[1] // scale
    sigma = sigma / scale
    imp = np.zeros((h, w), dtype=np.float64)
    pts = np.asarray(points, dtype=float).reshape(-1, 2) / scale
    if len(pts):
        xs = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        np.add.at(imp, (ys, xs), 1.0)
    return gaussian_filter(imp, sigma=sigma, mode="reflect")


def count_from_density(d: np.ndarray) -> float:
    return float(d.sum())


def foot_points(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre of each box: where a person touches the ground."""
    b = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)


def head_points(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Box centre, the usual anchor for crowd-counting density maps."""
    b = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def count_errors(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, mean signed error (bias) and mean relative error of per-frame counts."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    err = pred - true
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "mean_gt": float(true.mean()),
        "mape": float((np.abs(err) / np.maximum(true, 1)).mean()),
    }
