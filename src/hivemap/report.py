"""Run the full evaluation and write docs/results.md, results.json and figures."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from .data import in_ignore, xywh_to_xyxy  # noqa: E402
from .metrics import detection_ap  # noqa: E402
from .pipeline import (  # noqa: E402
    SOURCE_FPS,
    Config,
    Run,
    calibrate,
    dump_json,
    evaluate,
    ground,
    load,
)
from .tracker import iou_matrix  # noqa: E402

EVAL_SEQUENCES = [
    "uav0000086_00000_v",
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000339_00001_v",
]
HEIGHT_BINS = [(0, 40), (40, 70), (70, 110), (110, 1000)]


def threshold_sweep(runs: list[Run], grid=(0.3, 0.5, 0.7, 0.9)) -> dict[str, dict[str, float]]:
    """Pooled precision / recall / F1 over all evaluation frames at several score thresholds.

    Reported for transparency; the operating threshold itself is fixed a priori (0.5), not
    selected from this table.
    """
    gts, dets = [], []
    for run in runs:
        for (b, s), p in zip(run.dets, run.seq.frame_paths, strict=True):
            k = ~in_ignore(b, run.seq.ignore.get(int(p.stem), np.zeros((0, 4))))
            dets.append((b[k], s[k]))
            gts.append(xywh_to_xyxy(run.seq.gt[int(p.stem)][:, 1:5]))
    return {str(t): detection_ap(gts, dets, score_thr=t) for t in grid}


def recall_by_height(runs: list[Run]) -> dict[str, tuple[int, int]]:
    """Fraction of GT persons found (IoU >= 0.5 at the operating threshold) per pixel-height bin."""
    hit = np.zeros(len(HEIGHT_BINS), int)
    tot = np.zeros(len(HEIGHT_BINS), int)
    for run in runs:
        for (b, s), p in zip(run.dets, run.seq.frame_paths, strict=True):
            g = xywh_to_xyxy(run.seq.gt[int(p.stem)][:, 1:5])
            if not len(g):
                continue
            b = b[s >= run.cfg.det_thr]
            iou = iou_matrix(g, b)
            found = np.zeros(len(g), bool)
            if iou.size:
                r, c = linear_sum_assignment(-((iou >= 0.5) * iou))
                found[r[iou[r, c] >= 0.5]] = True
            h = g[:, 3] - g[:, 1]
            for k, (lo, hi) in enumerate(HEIGHT_BINS):
                m = (h >= lo) & (h < hi)
                tot[k] += m.sum()
                hit[k] += (found & m).sum()
    return {
        f"{lo}-{hi if hi < 1000 else 'inf'} px": (int(hit[k]), int(tot[k]))
        for k, (lo, hi) in enumerate(HEIGHT_BINS)
    }


def _mean(rows: list[dict], path: tuple[str, ...]) -> float:
    vals = []
    for r in rows:
        v = r
        for k in path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals))


def run_all(out: Path, stride: int, frames: int, backend: str = "frcnn") -> dict:
    base = Config(backend=backend, stride=stride, max_frames=frames)
    results: dict = {
        "config": {"stride": stride, "frames_per_sequence": frames, "backend": backend}
    }
    variants = {"full-frame": False, "tiled (640 px)": True}
    runs_by_variant: dict[str, list[Run]] = {}
    results["variants"] = {}
    for vname, tiled in variants.items():
        cfg = replace(base, tiled=tiled)
        thr = cfg.det_thr
        rows, runs = [], []
        for name in EVAL_SEQUENCES:
            run = load(name, cfg)
            runs.append(run)
            rows.append(evaluate(run))
            nocmc = evaluate(load(name, replace(cfg, cmc=False)))
            rows[-1]["mot_no_cmc"] = nocmc["mot"]
        runs_by_variant[vname] = runs
        results["variants"][vname] = {
            "det_thr": thr,
            "per_sequence": rows,
            "recall_by_height": recall_by_height(runs),
            "threshold_sweep": threshold_sweep(runs),
        }
    # Ground-plane analytics on the better variant (by mean AP).
    best = max(
        results["variants"],
        key=lambda v: _mean(results["variants"][v]["per_sequence"], ("detection", "ap")),
    )
    results["primary_variant"] = best
    results["ground"] = {}
    for run in runs_by_variant[best]:
        cam, diag = calibrate(run)
        g = ground(run, cam)
        results["ground"][run.seq.name] = {"camera": diag, **g.summary}
        figure_ground(run, g, out / f"ground_{run.seq.name}.png")
    figure_counts(runs_by_variant[best], out / "counts.png")
    figure_height_recall(results, out / "recall_by_height.png")
    dump_json(results, out / "results.json")
    write_markdown(results, out / "results.md")
    return results


def figure_counts(runs: list[Run], path: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(3.4 * len(runs), 2.8), sharey=False)
    for ax, run in zip(np.atleast_1d(axes), runs, strict=True):
        gt = [len(run.seq.gt[int(p.stem)]) for p in run.seq.frame_paths]
        pred = []
        for (b, s), p in zip(run.dets, run.seq.frame_paths, strict=True):
            k = ~in_ignore(b, run.seq.ignore.get(int(p.stem), np.zeros((0, 4))))
            pred.append(int(((s >= run.cfg.det_thr) & k).sum()))
        ax.plot(gt, label="ground truth", color="#222222")
        ax.plot(pred, label="detections", color="#d1495b")
        ax.set_title(run.seq.name.replace("uav0000", "").replace("_v", ""), fontsize=9)
        ax.set_xlabel("frame (sampled)")
    np.atleast_1d(axes)[0].set_ylabel("persons per frame")
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_height_recall(results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3))
    names = list(results["variants"])
    bins = list(results["variants"][names[0]]["recall_by_height"])
    w = 0.38
    for i, v in enumerate(names):
        rec = [h / max(n, 1) for h, n in results["variants"][v]["recall_by_height"].values()]
        ax.bar(
            np.arange(len(bins)) + (i - 0.5) * w, rec, w, label=v, color=["#7a7a7a", "#d1495b"][i]
        )
    counts = [n for _, n in results["variants"][names[0]]["recall_by_height"].values()]
    ax.set_xticks(
        range(len(bins)), [f"{b}\n(n={n})" for b, n in zip(bins, counts, strict=True)], fontsize=8
    )
    ax.set_ylabel("recall at operating threshold")
    ax.set_xlabel("ground-truth person height")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_ground(run: Run, g, path: Path) -> None:
    """Occupancy heatmap, flow field and peak-density series for one sequence."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ex, ey = g.edges_x, g.edges_y
    cx, cy = (ex[:-1] + ex[1:]) / 2, (ey[:-1] + ey[1:]) / 2
    im = axes[0].pcolormesh(ex, ey, g.occupancy, cmap="magma", shading="flat")
    axes[0].set_title("occupancy (persons per m$^2$, time-averaged)", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    sp = axes[1].pcolormesh(ex, ey, g.speed, cmap="viridis", shading="flat", vmin=0, vmax=2.5)
    X, Y = np.meshgrid(cx, cy)
    ok = np.isfinite(g.flow_u)
    mag = np.hypot(g.flow_u, g.flow_v)
    axes[1].quiver(
        X[ok],
        Y[ok],
        (g.flow_u / np.maximum(mag, 1e-6))[ok],
        (g.flow_v / np.maximum(mag, 1e-6))[ok],
        color="white",
        scale=25,
        width=0.004,
    )
    axes[1].set_title("mean flow direction and speed (m/s)", fontsize=9)
    fig.colorbar(sp, ax=axes[1], fraction=0.046)
    t = np.arange(len(g.peak_density)) * run.cfg.stride / SOURCE_FPS
    axes[2].plot(t, g.peak_density, color="#222222")
    axes[2].axhline(
        g.density_threshold,
        color="#d1495b",
        ls="--",
        label=f"alert threshold {g.density_threshold} /m$^2$",
    )
    axes[2].set_title("peak local density (2 m Gaussian window)", fontsize=9)
    axes[2].set_xlabel("time (s, assuming 30 fps source)")
    axes[2].legend(frameon=False, fontsize=8)
    for a in axes[:2]:
        a.set_xlabel("ground X (m)")
        a.set_ylabel("ground Z (m)")
        a.set_aspect("equal")
    fig.suptitle(run.seq.name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _rt(rows: list[dict], k: str) -> str:
    return _fmt(_mean(rows, ("runtime_ms", k)), 1)


def _fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def write_markdown(results: dict, path: Path) -> None:
    cfg = results["config"]
    L = ["# Results", ""]
    L.append(
        f"Subset: {len(EVAL_SEQUENCES)} VisDrone2019-MOT validation sequences, the first "
        f"{cfg['frames_per_sequence']} annotated frames of each at every {cfg['stride']}rd frame "
        f"(about {SOURCE_FPS / cfg['stride']:.0f} fps effective). Detector: torchvision Faster R-CNN "
        f"ResNet-50 FPN v2 (COCO). Person = VisDrone classes pedestrian + people. The score "
        f"threshold (0.5) and all tracker parameters were fixed before evaluation and not tuned."
    )
    L.append("")
    for vname, v in results["variants"].items():
        L += [f"## {vname} (score threshold {v['det_thr']})", ""]
        L.append(
            "| sequence | GT persons | AP@0.5 | precision | recall | MOTA | IDF1 | ID sw. | count MAE | count bias | fps |"
        )
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in v["per_sequence"]:
            d, m, c = r["detection"], r["mot"], r["counts"]["detections"]
            fps = 1000 / r["runtime_ms"]["total"]
            L.append(
                f"| {r['sequence']} | {d['n_gt']} | {_fmt(d['ap'])} | {_fmt(d['precision'])} | {_fmt(d['recall'])} | "
                f"{_fmt(m['mota'])} | {_fmt(m['idf1'])} | {m['id_switches']} | {_fmt(c['mae'], 2)} | {c['bias']:+.2f} | {_fmt(fps, 2)} |"
            )
        rows = v["per_sequence"]
        L.append(
            f"| **mean** | | {_fmt(_mean(rows, ('detection', 'ap')))} | {_fmt(_mean(rows, ('detection', 'precision')))} | "
            f"{_fmt(_mean(rows, ('detection', 'recall')))} | {_fmt(_mean(rows, ('mot', 'mota')))} | "
            f"{_fmt(_mean(rows, ('mot', 'idf1')))} | {sum(r['mot']['id_switches'] for r in rows)} (sum) | "
            f"{_fmt(_mean(rows, ('counts', 'detections', 'mae')), 2)} | {_mean(rows, ('counts', 'detections', 'bias')):+.2f} | "
            f"{_fmt(1000 / _mean(rows, ('runtime_ms', 'total')), 2)} |"
        )
        L.append("")
        L.append("Recall by ground-truth person height (pooled over sequences):")
        L.append("")
        L.append("| height | found / total | recall |")
        L.append("|---|---|---|")
        for k, (h, n) in v["recall_by_height"].items():
            L.append(f"| {k} | {h} / {n} | {_fmt(h / max(n, 1), 2)} |")
        L.append("")
        L.append("Score-threshold sensitivity (pooled over all evaluation frames):")
        L.append("")
        L.append("| threshold | precision | recall | F1 |")
        L.append("|---|---|---|---|")
        for t, d in v["threshold_sweep"].items():
            L.append(f"| {t} | {_fmt(d['precision'])} | {_fmt(d['recall'])} | {_fmt(d['f1'])} |")
        L.append("")
        L.append("Camera-motion compensation ablation (tracker input identical):")
        L.append("")
        L.append("| | MOTA | IDF1 | ID switches |")
        L.append("|---|---|---|---|")
        L.append(
            f"| with compensation | {_fmt(_mean(rows, ('mot', 'mota')))} | {_fmt(_mean(rows, ('mot', 'idf1')))} | {sum(r['mot']['id_switches'] for r in rows)} |"
        )
        L.append(
            f"| without | {_fmt(_mean(rows, ('mot_no_cmc', 'mota')))} | {_fmt(_mean(rows, ('mot_no_cmc', 'idf1')))} | {sum(r['mot_no_cmc']['id_switches'] for r in rows)} |"
        )
        L.append("")
        L.append("Per-frame runtime, milliseconds (mean over sequences):")
        L.append("")
        L.append("| detect | camera motion | track | density | total |")
        L.append("|---|---|---|---|---|")
        rt = partial(_rt, rows)
        L.append(
            f"| {rt('detect')} | {rt('camera_motion')} | {rt('track')} | {rt('density')} | {rt('total')} |"
        )
        L.append("")
        L.append(
            "Count error of the density map integral and of the confirmed tracks (MAE, mean over sequences):"
        )
        L.append("")
        L.append(
            f"detections {_fmt(_mean(rows, ('counts', 'detections', 'mae')), 2)}, "
            f"density integral {_fmt(_mean(rows, ('counts', 'density_integral', 'mae')), 2)}, "
            f"confirmed tracks {_fmt(_mean(rows, ('counts', 'tracks', 'mae')), 2)}; "
            f"mean GT count {_fmt(_mean(rows, ('counts', 'detections', 'mean_gt')), 1)} persons per frame."
        )
        L.append("")
    L += ["## Ground-plane analytics", "", f"Primary variant: {results['primary_variant']}.", ""]
    L.append(
        "| sequence | est. camera height (m) | est. pitch (deg) | median track speed (m/s) | p90 speed (m/s) | peak density (/m2) | alert frames |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for name, g in results["ground"].items():
        c = g["camera"]
        L.append(
            f"| {name} | {c['cam_height_m']:.1f} | {c['pitch_deg']:.0f} | {g['median_track_speed_ms']:.2f} | "
            f"{g['p90_track_speed_ms']:.2f} | {g['peak_density_max']:.2f} | {g['alert_frames']} / {g['frames']} |"
        )
    L += [
        "",
        "Figures: `counts.png`, `recall_by_height.png`, `ground_<sequence>.png` in this folder.",
        "",
    ]
    path.write_text("\n".join(L))
