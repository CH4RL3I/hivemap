"""Frame-to-frame camera motion estimation (sparse optical flow + robust similarity fit)."""

from __future__ import annotations

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0, 0], [0, 1.0, 0]])


def estimate_motion(prev_rgb: np.ndarray, rgb: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """2x3 similarity mapping ``prev`` pixel coordinates to ``rgb`` pixel coordinates.

    Moving people and vehicles are outliers to the dominant background motion, which RANSAC
    rejects. Falls back to identity when too few points are tracked.
    """
    a = cv2.cvtColor(cv2.resize(prev_rgb, None, fx=scale, fy=scale), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cv2.resize(rgb, None, fx=scale, fy=scale), cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if p0 is None or len(p0) < 10:
        return IDENTITY.copy()
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 10:
        return IDENTITY.copy()
    A, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if A is None:
        return IDENTITY.copy()
    A = A.astype(float)
    A[:, 2] /= scale  # translation back to full-resolution pixels
    return A


def compose(A_to_ref: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Given frame t-1 -> ref and step t-1 -> t, return frame t -> ref."""
    S = np.vstack([step, [0, 0, 1]])
    T = np.vstack([A_to_ref, [0, 0, 1]])
    return (T @ np.linalg.inv(S))[:2]


def warp_points(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    return p @ A[:, :2].T + A[:, 2]
