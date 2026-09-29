"""Person detection with pretrained, permissively licensed COCO detectors.

* ``frcnn``: torchvision Faster R-CNN ResNet-50 FPN v2 (code BSD-3; COCO weights).
* ``rtdetr``: RT-DETR R50 (Apache-2.0 code and weights, Hugging Face ``PekingU/rtdetr_r50vd``).

Drone pedestrians are small, so ``tile > 0`` also runs the detector on overlapping tiles that are
upscaled by the detector's own resize, and merges the results with the full-frame pass via NMS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

torch.set_num_threads(4)


def pick_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def tile_grid(w: int, h: int, tile: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Overlapping tile windows (x0, y0, x1, y1) covering a w x h image."""
    step = max(int(tile * (1 - overlap)), 1)
    xs = list(range(0, max(w - tile, 0) + 1, step))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    if xs[-1] + tile < w:
        xs.append(w - tile)
    if ys[-1] + tile < h:
        ys.append(h - tile)
    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in ys for x in xs]


class PersonDetector:
    def __init__(self, backend: str = "frcnn", tile: int = 0, device: torch.device | None = None):
        self.backend, self.tile = backend, tile
        self.device = device or pick_device()
        if backend == "frcnn":
            w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=w, box_score_thresh=0.05, box_detections_per_img=300
            )
            m.transform.min_size = (800,)
            self.model = m.eval().to(self.device)
            self.person_label = 1
        elif backend == "rtdetr":
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

            self.proc = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
            self.model = (
                RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
                .eval()
                .to(self.device)
            )
            self.person_label = 0
        else:
            raise ValueError(f"unknown backend {backend}")

    @torch.inference_mode()
    def _run(self, crops: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Detect persons in a list of HxWx3 uint8 RGB crops; boxes are xyxy in crop pixels."""
        out = []
        if self.backend == "frcnn":
            ts = [
                torch.from_numpy(np.ascontiguousarray(c).copy())
                .permute(2, 0, 1)
                .float()
                .div(255)
                .to(self.device)
                for c in crops
            ]
            for r in self.model(ts):
                k = r["labels"] == self.person_label
                out.append((r["boxes"][k].cpu().numpy(), r["scores"][k].cpu().numpy()))
        else:
            inputs = self.proc(images=[Image.fromarray(c) for c in crops], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            res = self.model(**inputs)
            sizes = torch.tensor([[c.shape[0], c.shape[1]] for c in crops])
            for r in self.proc.post_process_object_detection(
                res, threshold=0.05, target_sizes=sizes
            ):
                k = r["labels"] == self.person_label
                out.append((r["boxes"][k].cpu().numpy(), r["scores"][k].cpu().numpy()))
        return out

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (boxes xyxy, scores) of persons in a full frame (tiled if ``self.tile``)."""
        full = self.detect_full(rgb)
        return self.detect_tiled(rgb, full) if self.tile else full

    def detect_full(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b, s = self._run([rgb])[0]
        return b.reshape(-1, 4).astype(np.float32), s.astype(np.float32)

    def detect_tiled(
        self, rgb: np.ndarray, full: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merge a full-frame result with detections from overlapping tiles (NMS at IoU 0.5)."""
        h, w = rgb.shape[:2]
        wins = tile_grid(w, h, self.tile)
        all_b, all_s = [full[0]], [full[1]]
        for i in range(0, len(wins), 4):
            batch = wins[i : i + 4]
            for (x0, y0, x1, y1), (b, s) in zip(
                batch, self._run([rgb[y0:y1, x0:x1] for x0, y0, x1, y1 in batch]), strict=True
            ):
                if len(b) == 0:
                    continue
                # Drop boxes cut by an interior tile edge; the full pass or a neighbour sees them.
                cut = np.zeros(len(b), dtype=bool)
                if x0 > 0:
                    cut |= b[:, 0] < 2
                if y0 > 0:
                    cut |= b[:, 1] < 2
                if x1 < w:
                    cut |= b[:, 2] > (x1 - x0) - 2
                if y1 < h:
                    cut |= b[:, 3] > (y1 - y0) - 2
                all_b.append(b[~cut] + np.array([x0, y0, x0, y0]))
                all_s.append(s[~cut])
        boxes, scores = np.concatenate(all_b), np.concatenate(all_s)
        if len(boxes):
            keep = torchvision.ops.nms(
                torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), 0.5
            ).numpy()
            boxes, scores = boxes[keep], scores[keep]
        return boxes.reshape(-1, 4).astype(np.float32), scores.astype(np.float32)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))
