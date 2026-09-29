"""End-to-end pipeline for one sequence: detect, track, count, ground analytics, metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .analytics import GroundResult, analyse, ground_positions
from .camera import IDENTITY, compose, estimate_motion
from .data import Sequence, in_ignore, load_sequence, xywh_to_xyxy
from .density import count_errors, density_map, head_points
from .ground import Camera, fit_camera
from .metrics import detection_ap, evaluate_mot
from .tracker import ByteTracker

DATA_ROOT = Path("data/visdrone")
CACHE_ROOT = Path("data/cache")
WORK_WIDTH = 1344  # frames wider than this are shrunk before detection; boxes are mapped back
SOURCE_FPS = 30.0  # VisDrone videos are recorded at about 30 fps (assumption, not in the metadata)


@dataclass
class Config:
    backend: str = "frcnn"
    tile: int = 640  # tile size used when ``tiled`` is True
    tiled: bool = False
    stride: int = 2
    det_thr: float = 0.5
    cmc: bool = True
    max_frames: int | None = None


@dataclass
class Run:
    seq: Sequence
    cfg: Config
    dets: list[tuple[np.ndarray, np.ndarray]]
    det_ms: np.ndarray
    steps: list[np.ndarray]  # camera motion step t-1 -> t (identity for the first frame)
    cmc_ms: np.ndarray
    tracks: list[list[tuple[int, np.ndarray, float]]] = field(default_factory=list)
    track_ms: float = 0.0
    to_ref: list[np.ndarray] = field(default_factory=list)
    frame_shape: tuple[int, int] = (756, 1344)


def _cache_path(seq: Sequence, cfg: Config) -> Path:
    n = len(seq.frame_paths)
    return CACHE_ROOT / f"{seq.name}_{cfg.backend}_t{cfg.tile}_s{cfg.stride}_n{n}.npz"


def _variant(z, cfg: Config) -> tuple[list, np.ndarray]:
    """Pick the full-frame or tiled detections (and their runtime) from a cache file."""
    if cfg.tiled:
        dets = [(b, s) for b, s in zip(z["boxes_tile"], z["scores_tile"], strict=True)]
        return dets, z["full_ms"] + z["tile_ms"]
    dets = [(b, s) for b, s in zip(z["boxes_full"], z["scores_full"], strict=True)]
    return dets, z["full_ms"]


def compute_detections(seq: Sequence, cfg: Config, verbose: bool = True) -> Run:
    """Run the detector (and camera-motion estimation) on every frame, with an on-disk cache.

    One pass yields both the full-frame and the tiled result (the tiled merge reuses the
    full-frame detections), and their runtimes are recorded separately: the full-frame variant
    costs ``full_ms``, the tiled variant ``full_ms + tile_ms``.
    """
    cp = _cache_path(seq, cfg)
    if not cp.exists():
        from .detect import PersonDetector, load_rgb

        det = PersonDetector(cfg.backend, cfg.tile)
        full, tiled, full_ms, tile_ms, steps, cmc_ms = [], [], [], [], [], []
        prev = None
        for i, p in enumerate(seq.frame_paths):
            rgb, sc = load_rgb(p, WORK_WIDTH)
            t0 = time.perf_counter()
            f = det.detect_full(rgb)
            t1 = time.perf_counter()
            t = det.detect_tiled(rgb, f)
            t2 = time.perf_counter()
            f, t = (f[0] / sc, f[1]), (t[0] / sc, t[1])
            full.append(f)
            tiled.append(t)
            full_ms.append((t1 - t0) * 1000)
            tile_ms.append((t2 - t1) * 1000)
            t0 = time.perf_counter()
            step = IDENTITY.copy() if prev is None else estimate_motion(prev, rgb)
            step[:, 2] /= sc  # translation back to original-resolution pixels
            steps.append(step)
            cmc_ms.append((time.perf_counter() - t0) * 1000)
            prev = rgb
            if verbose and i % 10 == 0:
                print(
                    f"  {seq.name}: {i}/{len(seq.frame_paths)} full {np.mean(full_ms):.0f} ms,"
                    f" tiles {np.mean(tile_ms):.0f} ms",
                    flush=True,
                )

        def obj(lst):
            arr = np.empty(len(lst), dtype=object)
            for k, v in enumerate(lst):
                arr[k] = v
            return arr

        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cp,
            boxes_full=obj([d[0] for d in full]),
            scores_full=obj([d[1] for d in full]),
            boxes_tile=obj([d[0] for d in tiled]),
            scores_tile=obj([d[1] for d in tiled]),
            full_ms=np.array(full_ms),
            tile_ms=np.array(tile_ms),
            steps=np.array(steps),
            cmc_ms=np.array(cmc_ms),
        )
    z = np.load(cp, allow_pickle=True)
    dets, det_ms = _variant(z, cfg)
    w, h = Image.open(seq.frame_paths[0]).size
    return Run(seq, cfg, dets, det_ms, list(z["steps"]), z["cmc_ms"], frame_shape=(h, w))


def track(run: Run, **tracker_kwargs) -> Run:
    """Run the tracker over cached detections; fills ``tracks`` and ``to_ref``."""
    cfg = run.cfg
    kw = dict(high_thr=cfg.det_thr, new_thr=cfg.det_thr + 0.1, low_thr=0.1)
    kw.update(tracker_kwargs)
    # ``max_age`` is defined in source frames; convert to processed frames.
    kw.setdefault("max_age", max(int(round(30 / cfg.stride)), 2))
    tr = ByteTracker(**kw)
    run.tracks, run.to_ref = [], []
    A = IDENTITY.copy()
    t0 = time.perf_counter()
    for i, ((b, s), (p, step)) in enumerate(
        zip(run.dets, zip(run.seq.frame_paths, run.steps, strict=True), strict=True)
    ):
        if i > 0:
            A = compose(A, step)
        run.to_ref.append(A.copy())
        ign = run.seq.ignore.get(int(p.stem), np.zeros((0, 4)))
        keep = ~in_ignore(b, ign)
        cam = step if (cfg.cmc and i > 0) else None
        run.tracks.append(tr.update(b[keep], s[keep], cam))
    run.track_ms = (time.perf_counter() - t0) * 1000 / max(len(run.dets), 1)
    return run


def _gt_frames(run: Run) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for p in run.seq.frame_paths:
        g = run.seq.gt[int(p.stem)]
        out.append((g[:, 0].astype(int), xywh_to_xyxy(g[:, 1:5])))
    return out


def evaluate(run: Run) -> dict:
    """Detection AP, tracking metrics, count error and per-frame runtime for a tracked run."""
    gt = _gt_frames(run)
    ign = [run.seq.ignore.get(int(p.stem), np.zeros((0, 4))) for p in run.seq.frame_paths]
    dets = []
    for (b, s), ig in zip(run.dets, ign, strict=True):
        k = ~in_ignore(b, ig)
        dets.append((b[k], s[k]))
    det = detection_ap([g[1] for g in gt], dets, score_thr=run.cfg.det_thr)

    hyp = []
    for tr, ig in zip(run.tracks, ign, strict=True):
        if tr:
            ids = np.array([t[0] for t in tr])
            bx = np.array([t[1] for t in tr])
            k = ~in_ignore(bx, ig)
            hyp.append((ids[k], bx[k]))
        else:
            hyp.append((np.zeros(0, int), np.zeros((0, 4))))
    mot = evaluate_mot(gt, hyp)

    frame_shape = run.frame_shape
    t0 = time.perf_counter()
    pred_counts, dens_counts = [], []
    for b, s in dets:
        kept = b[s >= run.cfg.det_thr]
        pred_counts.append(len(kept))
        dens_counts.append(
            density_map(
                head_points(kept), frame_shape, sigma=15 * frame_shape[0] / 756, scale=4
            ).sum()
        )
    dens_ms = (time.perf_counter() - t0) * 1000 / max(len(dets), 1)
    true_counts = [len(g[0]) for g in gt]
    track_counts = [len(h[0]) for h in hyp]
    counts = {
        "detections": count_errors(pred_counts, true_counts),
        "density_integral": count_errors(dens_counts, true_counts),
        "tracks": count_errors(track_counts, true_counts),
    }
    det_ms = float(run.det_ms.mean())
    cmc_ms = float(run.cmc_ms.mean()) if run.cfg.cmc else 0.0
    return {
        "sequence": run.seq.name,
        "frames": len(gt),
        "detection": det,
        "mot": mot.as_dict(),
        "counts": counts,
        "runtime_ms": {
            "detect": det_ms,
            "camera_motion": cmc_ms,
            "track": run.track_ms,
            "density": dens_ms,
            "total": det_ms + cmc_ms + run.track_ms + dens_ms,
        },
    }


def calibrate(run: Run) -> tuple[Camera, dict]:
    """Fit the ground-plane camera from confident detections (no ground truth is used)."""
    boxes = [b[s >= 0.7] for b, s in run.dets if len(b)]
    allb = np.concatenate([b for b in boxes if len(b)]) if boxes else np.zeros((0, 4))
    # Stationary or seated people and very small boxes bias the fit; keep plausible pedestrians.
    allb = allb[(allb[:, 3] - allb[:, 1]) > 12]
    return fit_camera(allb, run.frame_shape[1], run.frame_shape[0])


def ground(run: Run, camera: Camera, **kw) -> GroundResult:
    frames = [[(t[0], t[1]) for t in tr] for tr in run.tracks]
    pos = ground_positions(frames, camera, run.to_ref)
    dt = run.cfg.stride / SOURCE_FPS
    return analyse(pos, dt, **kw)


def write_mot(run: Run, path: Path) -> None:
    """Write tracks in MOTChallenge text format (frame, id, x, y, w, h, score, -1, -1, -1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p, tr in zip(run.seq.frame_paths, run.tracks, strict=True):
            for tid, b, s in tr:
                f.write(
                    f"{int(p.stem)},{tid},{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},"
                    f"{b[3] - b[1]:.1f},{s:.3f},-1,-1,-1\n"
                )


def load(name: str, cfg: Config, root: Path = DATA_ROOT) -> Run:
    seq = load_sequence(root, name, stride=cfg.stride, max_frames=cfg.max_frames)
    return track(compute_detections(seq, cfg))


def dump_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
