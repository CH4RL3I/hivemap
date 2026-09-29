"""Annotated GIF: track boxes, trails and a density overlay."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .density import density_map, head_points
from .pipeline import Run


def _colour(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid * 7919)
    c = rng.integers(60, 256, 3)
    return int(c[0]), int(c[1]), int(c[2])


def render_frames(
    run: Run, width: int = 640, trail: int = 12, background: str = "dark"
) -> list[np.ndarray]:
    """Render one image per frame. ``background='dark'`` draws on an empty canvas so that no
    VisDrone pixels are redistributed; ``'video'`` overlays on the source frame (local use)."""
    frames = []
    history: dict[int, list[tuple[int, np.ndarray]]] = {}  # id -> [(frame idx, foot in ref coords)]
    for i, (p, tracks) in enumerate(zip(run.seq.frame_paths, run.tracks, strict=True)):
        if background == "video":
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
        else:
            h, w = run.frame_shape
            img = np.full((h, w, 3), 18, np.uint8)
            img[::96, :] = 34
            img[:, ::96] = 34
        A = np.vstack([run.to_ref[i], [0, 0, 1]])
        Ainv = np.linalg.inv(A)
        boxes = np.array([t[1] for t in tracks]).reshape(-1, 4)
        for tid, b, _ in tracks:
            foot = np.array([(b[0] + b[2]) / 2, b[3], 1.0])
            history.setdefault(tid, []).append((i, (A @ foot)[:2]))

        # Density overlay: red-yellow heat, transparent where density is negligible.
        d = density_map(head_points(boxes), (h, w), sigma=max(h / 40, 10), scale=4)
        count = d.sum()
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        peak = max(d.max(), 1e-9)
        norm = np.clip(d / peak, 0, 1)
        heat = cv2.cvtColor(
            cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
            cv2.COLOR_BGR2RGB,
        )
        alpha = (np.clip(norm * 1.6, 0, 0.55))[..., None]
        img = (img * (1 - alpha) + heat * alpha).astype(np.uint8)

        for tid, b, _ in tracks:
            col = _colour(tid)
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
            pts = [(f, q) for f, q in history[tid] if i - f < trail]
            if len(pts) > 1:
                cur = np.array(
                    [(Ainv @ np.array([q[0], q[1], 1.0]))[:2] for _, q in pts], dtype=np.int32
                )
                cv2.polylines(img, [cur.reshape(-1, 1, 2)], False, col, 2)
        s = width / w
        img = cv2.resize(img, (width, int(h * s)), interpolation=cv2.INTER_AREA)
        label = f"{run.seq.name}  frame {int(p.stem)}  tracked: {len(tracks)}  density integral: {count:.1f}"
        cv2.rectangle(img, (0, 0), (width, 20), (0, 0, 0), -1)
        cv2.putText(
            img, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        frames.append(img)
    return frames


def write_gif(frames: list[np.ndarray], path: Path, fps: int = 8, max_mb: float = 5.0) -> Path:
    """Write a looping GIF, shrinking colours and then size until it is under ``max_mb``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scale, colours = 1.0, 128
    for _ in range(8):
        ims = []
        for f in frames:
            im = Image.fromarray(f)
            if scale < 1:
                im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
            ims.append(
                im.quantize(
                    colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
                )
            )
        ims[0].save(
            path,
            save_all=True,
            append_images=ims[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        if path.stat().st_size <= max_mb * 1024 * 1024 * 0.9:
            return path
        scale *= 0.85
        colours = max(colours - 16, 48)
    return path
