"""Constant-velocity Kalman filter over a bounding box (cx, cy, w, h) and its velocities."""

from __future__ import annotations

import numpy as np


class KalmanBoxFilter:
    """State x = [cx, cy, w, h, vcx, vcy, vw, vh]; measurement z = [cx, cy, w, h].

    Process and measurement noise scale with the box size (as in DeepSORT/ByteTrack), so the
    same settings work for large and tiny pedestrians.
    """

    def __init__(self, z: np.ndarray, std_pos: float = 1 / 20, std_vel: float = 1 / 160) -> None:
        self.std_pos, self.std_vel = std_pos, std_vel
        self.x = np.concatenate([np.asarray(z, dtype=float), np.zeros(4)])
        h = z[3]
        std = np.array([2 * std_pos * h] * 2 + [2 * std_pos * h] * 2 + [10 * std_vel * h] * 4)
        self.P = np.diag(std**2)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def _q(self) -> np.ndarray:
        h = max(self.x[3], 1.0)
        std = np.array([self.std_pos * h] * 4 + [self.std_vel * h] * 4)
        return np.diag(std**2)

    def _r(self) -> np.ndarray:
        h = max(self.x[3], 1.0)
        return np.diag((np.array([self.std_pos * h] * 4)) ** 2)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._q()
        return self.x[:4].copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        S = self.H @ self.P @ self.H.T + self._r()
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, dtype=float) - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self.x[:4].copy()

    def apply_affine(self, A: np.ndarray) -> None:
        """Warp the state by a 2x3 affine (camera-motion compensation)."""
        M, t = A[:, :2], A[:, 2]
        self.x[:2] = M @ self.x[:2] + t
        self.x[4:6] = M @ self.x[4:6]
        scale = np.sqrt(abs(np.linalg.det(M)))
        self.x[2:4] *= scale
        self.x[6:8] *= scale
