"""Ground-plane geometry: an image-to-metres homography from a pinhole camera model.

Assumptions (all documented in the README, none of them measured):

* the scene is one flat ground plane;
* the camera is a pinhole with zero roll and a horizontal field of view ``hfov`` (default 80
  degrees, typical for consumer drones; VisDrone ships no intrinsics);
* an adult is 1.7 m tall, and detected person boxes span head to foot.

Camera pitch and height are then fitted so that the predicted pixel height of a 1.7 m person at
each foot position matches the detected box heights (``fit_camera``). The ground homography
follows from the intrinsics, pitch and height (``Camera.homography``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PERSON_HEIGHT_M = 1.7


@dataclass
class Camera:
    width: int
    height: int
    f: float  # focal length in pixels
    pitch: float  # radians below the horizontal
    cam_height: float  # metres above the ground plane

    def homography(self) -> np.ndarray:
        """3x3 matrix mapping homogeneous image pixels to ground (X right, Z forward) metres."""
        K = np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1.0]])
        c, s = np.cos(self.pitch), np.sin(self.pitch)
        R = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])  # camera frame -> level frame
        M = R @ np.linalg.inv(K)
        # ray = M @ [x, y, 1]; ground hit at t = Hc / ray_y; X = t*ray_x, Z = t*ray_z
        return np.array([M[0], M[2], M[1] / self.cam_height])


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def _level_rotation(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])  # camera frame -> level frame


def predict_heights(
    cam: Camera, feet: np.ndarray, person_height: float = PERSON_HEIGHT_M
) -> np.ndarray:
    """Pixel height of an upright person of ``person_height`` metres standing at image ``feet``.

    The foot pixel is back-projected to the ground, lifted by the person's height and projected
    again. Feet at or above the horizon (no ground intersection) give NaN.
    """
    feet = np.asarray(feet, dtype=float).reshape(-1, 2)
    K = np.array([[cam.f, 0, cam.width / 2], [0, cam.f, cam.height / 2], [0, 0, 1.0]])
    R = _level_rotation(cam.pitch)
    ray = np.c_[feet, np.ones(len(feet))] @ (R @ np.linalg.inv(K)).T
    ok = ray[:, 1] > 1e-6
    t = np.where(ok, cam.cam_height / np.where(ok, ray[:, 1], 1.0), np.nan)
    ground = ray * t[:, None]  # level-frame point (X, Hc, Z)
    head = ground.copy()
    head[:, 1] -= person_height
    hc = head @ R  # level -> camera frame (R.T applied to row vectors)
    y_head = cam.f * hc[:, 1] / hc[:, 2] + cam.height / 2
    return feet[:, 1] - y_head


def fit_camera(
    boxes: np.ndarray,
    width: int,
    height: int,
    hfov_deg: float = 80.0,
    person_height: float = PERSON_HEIGHT_M,
    max_boxes: int = 600,
) -> tuple[Camera, dict[str, float]]:
    """Estimate camera pitch and height from person boxes (xyxy) gathered over a sequence.

    Robust least squares (soft-L1 on log box height) over the two unknowns, started from several
    pitches. Only the *scale* of the scene is observable this way, so the result is approximate;
    the README quantifies how much that matters.
    """
    from scipy.optimize import least_squares

    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(b) > max_boxes:
        b = b[np.random.default_rng(0).choice(len(b), max_boxes, replace=False)]
    feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
    hobs = np.maximum(b[:, 3] - b[:, 1], 1.0)
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)

    def resid(theta: np.ndarray) -> np.ndarray:
        cam = Camera(width, height, f, float(theta[0]), float(np.exp(theta[1])))
        hp = predict_heights(cam, feet, person_height)
        r = np.log(np.where(np.isfinite(hp) & (hp > 0.5), hp, 1e-3)) - np.log(hobs)
        return np.clip(r, -5, 5)

    best = None
    for p0 in (25, 45, 65, 80):
        h0 = person_height * f / max(np.median(hobs), 1.0) * np.sin(np.radians(p0))
        sol = least_squares(
            resid,
            x0=[np.radians(p0), np.log(np.clip(h0, 3, 200))],
            bounds=([np.radians(8), np.log(2.0)], [np.radians(88), np.log(300.0)]),
            loss="soft_l1",
            f_scale=0.15,
        )
        if best is None or sol.cost < best.cost:
            best = sol
    cam = Camera(width, height, float(f), float(best.x[0]), float(np.exp(best.x[1])))
    r = resid(best.x)
    diag = {
        "cam_height_m": cam.cam_height,
        "pitch_deg": float(np.degrees(cam.pitch)),
        "focal_px": cam.f,
        "n_used": int(len(b)),
        "resid_log_median_abs": float(np.median(np.abs(r))),
    }
    return cam, diag


def grid_edges(xy: np.ndarray, cell: float, pad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.percentile(xy, 1, axis=0) - pad, np.percentile(xy, 99, axis=0) + pad
    return (
        np.arange(np.floor(lo[0] / cell) * cell, hi[0] + cell, cell),
        np.arange(np.floor(lo[1] / cell) * cell, hi[1] + cell, cell),
    )
