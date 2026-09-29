"""Ground-plane analytics: occupancy, flow field, speeds and density alerts, in metres."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .camera import warp_points
from .ground import Camera, apply_h, grid_edges


@dataclass
class GroundResult:
    edges_x: np.ndarray
    edges_y: np.ndarray
    occupancy: np.ndarray  # mean persons per m^2 over the sequence, shape (ny, nx)
    flow_u: np.ndarray  # mean velocity east (m/s) per cell, NaN where too few samples
    flow_v: np.ndarray
    speed: np.ndarray  # mean speed (m/s) per cell
    track_speeds: dict[int, float]  # mean speed of each long-enough track
    peak_density: np.ndarray  # per frame, max local density (persons/m^2)
    alert_frames: list[int]
    density_threshold: float
    summary: dict[str, float]


def ground_positions(
    frames: list[list[tuple[int, np.ndarray]]],
    camera: Camera,
    to_ref: list[np.ndarray],
) -> list[dict[int, np.ndarray]]:
    """Per frame, {track id: ground (X, Z) in metres} of each track's foot point.

    ``to_ref[t]`` is the 2x3 affine taking frame t pixels into frame-0 pixels (camera motion), so
    a fixed ground plane can be used although the drone moves.
    """
    H = camera.homography()
    out = []
    for tracks, A in zip(frames, to_ref, strict=True):
        d = {}
        if tracks:
            ids = [t for t, _ in tracks]
            b = np.array([bx for _, bx in tracks])
            feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
            g = apply_h(H, warp_points(A, feet))
            d = dict(zip(ids, g, strict=True))
        out.append(d)
    return out


def analyse(
    pos: list[dict[int, np.ndarray]],
    dt: float,
    cell: float = 2.0,
    density_threshold: float = 0.5,
    kernel_sigma_m: float = 2.0,
    min_track_s: float = 1.0,
    min_samples: int = 5,
) -> GroundResult:
    """Compute occupancy, flow and alerts from per-frame ground positions.

    ``dt`` is the time between consecutive processed frames in seconds. Velocity uses a central
    difference over three frames of each track to suppress detector jitter.
    """
    allxy = np.array([p for d in pos for p in d.values()])
    ex, ey = grid_edges(allxy, cell, pad=cell)
    nx, ny = len(ex) - 1, len(ey) - 1
    occ = np.zeros((ny, nx))
    sum_u = np.zeros((ny, nx))
    sum_v = np.zeros((ny, nx))
    sum_s = np.zeros((ny, nx))
    n = np.zeros((ny, nx))

    traj: dict[int, list[tuple[int, np.ndarray]]] = {}
    for f, d in enumerate(pos):
        for t, p in d.items():
            traj.setdefault(t, []).append((f, p))

    track_speeds: dict[int, float] = {}
    for t, pts in traj.items():
        if len(pts) < 3:
            continue
        fr = np.array([f for f, _ in pts])
        xy = np.array([p for _, p in pts])
        # Only use consecutive-frame samples for velocity.
        speeds = []
        for i in range(1, len(pts) - 1):
            if fr[i + 1] - fr[i - 1] != 2:
                continue
            v = (xy[i + 1] - xy[i - 1]) / (2 * dt)
            speeds.append(np.linalg.norm(v))
            ix = np.searchsorted(ex, xy[i, 0], side="right") - 1
            iy = np.searchsorted(ey, xy[i, 1], side="right") - 1
            if 0 <= ix < nx and 0 <= iy < ny:
                sum_u[iy, ix] += v[0]
                sum_v[iy, ix] += v[1]
                sum_s[iy, ix] += speeds[-1]
                n[iy, ix] += 1
        if len(pts) * dt >= min_track_s and speeds:
            track_speeds[t] = float(np.mean(speeds))

    peak = np.zeros(len(pos))
    alerts = []
    fine = cell / 2
    fx = np.arange(ex[0], ex[-1] + fine, fine)
    fy = np.arange(ey[0], ey[-1] + fine, fine)
    for f, d in enumerate(pos):
        if not d:
            continue
        xy = np.array(list(d.values()))
        h, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[fy, fx])
        occ += np.histogram2d(xy[:, 1], xy[:, 0], bins=[ey, ex])[0]
        # Persons per m^2: Gaussian-smoothed counts (mass preserved) over fine^2 m^2 cells.
        dens = gaussian_filter(h, sigma=kernel_sigma_m / fine, mode="constant") / fine**2
        peak[f] = dens.max()
        if peak[f] > density_threshold:
            alerts.append(f)
    occ = occ / max(len(pos), 1) / cell**2

    with np.errstate(invalid="ignore", divide="ignore"):
        ok = n >= min_samples
        u = np.where(ok, sum_u / n, np.nan)
        v = np.where(ok, sum_v / n, np.nan)
        s = np.where(ok, sum_s / n, np.nan)
    sp = np.array(list(track_speeds.values())) if track_speeds else np.array([np.nan])
    summary = {
        "n_tracks_speed": len(track_speeds),
        "median_track_speed_ms": float(np.nanmedian(sp)),
        "p90_track_speed_ms": float(np.nanpercentile(sp, 90)),
        "peak_density_max": float(peak.max()),
        "peak_density_median": float(np.median(peak)),
        "alert_frames": len(alerts),
        "frames": len(pos),
        "extent_x_m": float(ex[-1] - ex[0]),
        "extent_y_m": float(ey[-1] - ey[0]),
    }
    return GroundResult(
        ex, ey, occ, u, v, s, track_speeds, peak, alerts, density_threshold, summary
    )
