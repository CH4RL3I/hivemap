"""ByteTrack-style multi-object tracker: Kalman filter + Hungarian matching on IoU.

Two-stage association: confident detections are matched to all live tracks first; the remaining
tracks are then matched against low-confidence detections, which recovers occluded people
without opening the door to background clutter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import KalmanBoxFilter


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes, shape (len(a), len(b))."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def associate(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Minimum-cost assignment with a gate: pairs costlier than ``max_cost`` are rejected."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if cost[r, c] <= max_cost]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    return (
        matches,
        [i for i in range(n) if i not in matched_r],
        [j for j in range(m) if j not in matched_c],
    )


def xyxy_to_z(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], dtype=float)


def z_to_xyxy(z: np.ndarray) -> np.ndarray:
    return np.array([z[0] - z[2] / 2, z[1] - z[3] / 2, z[0] + z[2] / 2, z[1] + z[3] / 2])


@dataclass
class Track:
    tid: int
    kf: KalmanBoxFilter
    hits: int = 1
    age_since_update: int = 0
    score: float = 1.0

    @property
    def box(self) -> np.ndarray:
        return z_to_xyxy(self.kf.x[:4])


class ByteTracker:
    def __init__(
        self,
        high_thr: float = 0.5,
        low_thr: float = 0.1,
        new_thr: float = 0.6,
        match_iou: float = 0.3,
        second_iou: float = 0.4,
        max_age: int = 20,
        min_hits: int = 2,
    ) -> None:
        self.high_thr, self.low_thr, self.new_thr = high_thr, low_thr, new_thr
        self.match_iou, self.second_iou = match_iou, second_iou
        self.max_age, self.min_hits = max_age, min_hits
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = 0

    def update(
        self, boxes: np.ndarray, scores: np.ndarray, camera: np.ndarray | None = None
    ) -> list[tuple[int, np.ndarray, float]]:
        """Advance one frame. ``camera`` is an optional 2x3 affine mapping the previous frame's
        coordinates into this frame's. Returns (track id, xyxy box, score) for confirmed tracks."""
        self.frame += 1
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        for t in self.tracks:
            if camera is not None:
                t.kf.apply_affine(camera)
            t.kf.predict()
            t.age_since_update += 1

        high = scores >= self.high_thr
        low = (scores >= self.low_thr) & ~high
        d_hi, s_hi = boxes[high], scores[high]
        d_lo, s_lo = boxes[low], scores[low]

        # Stage 1: confident detections against every live track.
        pred = np.array([t.box for t in self.tracks]).reshape(-1, 4)
        cost = 1.0 - iou_matrix(pred, d_hi)
        m1, un_t, un_d = associate(cost, 1.0 - self.match_iou)
        for ti, di in m1:
            self._update(self.tracks[ti], d_hi[di], s_hi[di])

        # Stage 2: leftover tracks (that were tracked recently) against low-confidence detections.
        rest = [i for i in un_t if self.tracks[i].age_since_update <= 2]
        pred2 = np.array([self.tracks[i].box for i in rest]).reshape(-1, 4)
        m2, _, _ = associate(1.0 - iou_matrix(pred2, d_lo), 1.0 - self.second_iou)
        for ri, di in m2:
            self._update(self.tracks[rest[ri]], d_lo[di], s_lo[di])

        # New tracks from unmatched confident detections.
        for di in un_d:
            if s_hi[di] >= self.new_thr:
                self._spawn(d_hi[di], s_hi[di])

        self.tracks = [t for t in self.tracks if t.age_since_update <= self.max_age]
        return [
            (t.tid, t.box, t.score)
            for t in self.tracks
            if t.age_since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits)
        ]

    def _update(self, t: Track, box: np.ndarray, score: float) -> None:
        t.kf.update(xyxy_to_z(box))
        t.hits += 1
        t.age_since_update = 0
        t.score = float(score)

    def _spawn(self, box: np.ndarray, score: float) -> None:
        self.tracks.append(
            Track(self._next_id, KalmanBoxFilter(xyxy_to_z(box)), score=float(score))
        )
        self._next_id += 1
