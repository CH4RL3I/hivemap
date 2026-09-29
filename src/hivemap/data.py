"""VisDrone-MOT sequence loading and ground-truth parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PERSON_CLASSES = (1, 2)  # pedestrian, people (a person on a bike/other is not included here)
IGNORE_CLASS = 0  # ignored regions


@dataclass
class Sequence:
    name: str
    frame_paths: list[Path]
    # frame number (1-based, as in the annotation file) -> (N, 5) array [id, x, y, w, h]
    gt: dict[int, np.ndarray] = field(default_factory=dict)
    ignore: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> (M, 4) [x, y, w, h]

    def frame_numbers(self) -> list[int]:
        return [int(p.stem) for p in self.frame_paths]


def parse_annotations(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Parse a VisDrone-MOT annotation file into person GT and ignore regions per frame.

    Row format: frame, id, x, y, w, h, score, category, truncation, occlusion.
    Only rows with score 1 and a person category count as ground truth.
    """
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    gt: dict[int, np.ndarray] = {}
    ign: dict[int, np.ndarray] = {}
    person = np.isin(rows[:, 7], PERSON_CLASSES) & (rows[:, 6] == 1)
    for fr in np.unique(rows[:, 0]).astype(int):
        r = rows[rows[:, 0] == fr]
        p = person[rows[:, 0] == fr]
        gt[fr] = r[p][:, [1, 2, 3, 4, 5]].copy()
        ign[fr] = r[r[:, 7] == IGNORE_CLASS][:, 2:6].copy()
    return gt, ign


def load_sequence(
    root: Path, name: str, stride: int = 1, max_frames: int | None = None
) -> Sequence:
    """Load frames and annotations for ``name`` from ``root`` (as written by ``hivemap fetch``).

    Only frames that carry annotations are used, so that every evaluated frame has ground truth.
    """
    gt, ign = parse_annotations(root / "annotations" / f"{name}.txt")
    paths = sorted((root / "sequences" / name).glob("*.jpg"))
    paths = [p for p in paths if int(p.stem) in gt]
    paths = paths[::stride]
    if max_frames:
        paths = paths[:max_frames]
    return Sequence(name, paths, gt, ign)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def in_ignore(boxes_xyxy: np.ndarray, ignore_xywh: np.ndarray, frac: float = 0.5) -> np.ndarray:
    """Boolean mask of boxes whose area lies at least ``frac`` inside an ignored region."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    mask = np.zeros(len(boxes_xyxy), dtype=bool)
    if len(ignore_xywh) == 0 or len(boxes_xyxy) == 0:
        return mask
    ig = xywh_to_xyxy(ignore_xywh)
    area = np.maximum(
        (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]), 1e-9
    )
    for r in ig:
        iw = np.clip(
            np.minimum(boxes_xyxy[:, 2], r[2]) - np.maximum(boxes_xyxy[:, 0], r[0]), 0, None
        )
        ih = np.clip(
            np.minimum(boxes_xyxy[:, 3], r[3]) - np.maximum(boxes_xyxy[:, 1], r[1]), 0, None
        )
        mask |= (iw * ih) / area >= frac
    return mask
