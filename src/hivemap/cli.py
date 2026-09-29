"""Command line interface: fetch, run, eval, render, bench."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, help="Aerial crowd analysis from drone video.")

Stride = Annotated[int, typer.Option(help="Use every n-th annotated frame.")]
Frames = Annotated[int, typer.Option(help="Frames per sequence after striding.")]


@app.command()
def fetch(
    sequence: Annotated[
        list[str] | None, typer.Argument(help="Sequence names (default subset).")
    ] = None,
    list_all: Annotated[
        bool, typer.Option("--list", help="List validation sequences and exit.")
    ] = False,
    max_frames: Annotated[
        int | None, typer.Option(help="Only download the first n frames.")
    ] = None,
    dest: Path = Path("data/visdrone"),
) -> None:
    """Download a subset of VisDrone2019-MOT-val into data/ (about 150 MB for the default subset)."""
    from .fetch import fetch as do_fetch
    from .fetch import list_sequences
    from .report import EVAL_SEQUENCES

    if list_all:
        for s in list_sequences():
            typer.echo(s)
        return
    names = sequence or EVAL_SEQUENCES
    do_fetch(names, dest, max_frames=max_frames if sequence else None)
    typer.echo(f"fetched {len(names)} sequences into {dest}")


@app.command()
def run(
    sequence: str,
    tiled: Annotated[bool, typer.Option(help="Tiled inference for small people.")] = True,
    stride: Stride = 3,
    frames: Frames = 70,
    det_thr: Annotated[float, typer.Option(help="Detection score threshold.")] = 0.5,
    backend: Annotated[str, typer.Option(help="frcnn or rtdetr")] = "frcnn",
    out: Path = Path("runs"),
) -> None:
    """Detect, track and analyse one sequence; write tracks, metrics and ground figures to runs/."""
    from .pipeline import Config, calibrate, dump_json, evaluate, ground, load, write_mot
    from .report import figure_ground

    cfg = Config(backend=backend, tiled=tiled, stride=stride, max_frames=frames, det_thr=det_thr)
    r = load(sequence, cfg)
    res = evaluate(r)
    cam, diag = calibrate(r)
    g = ground(r, cam)
    res["ground"] = {"camera": diag, **g.summary}
    d = out / sequence
    write_mot(r, d / "tracks.txt")
    dump_json(res, d / "summary.json")
    figure_ground(r, g, d / "ground.png")
    m = res["mot"]
    typer.echo(
        f"{sequence}: AP50 {res['detection']['ap']:.3f}  MOTA {m['mota']:.3f}  IDF1 {m['idf1']:.3f}  "
        f"ID switches {m['id_switches']}  count MAE {res['counts']['detections']['mae']:.2f}  "
        f"{res['runtime_ms']['total']:.0f} ms/frame"
    )
    typer.echo(f"wrote {d}")


@app.command("eval")
def eval_(
    stride: Stride = 3,
    frames: Frames = 70,
    backend: str = "frcnn",
    out: Path = Path("docs"),
) -> None:
    """Evaluate full-frame vs tiled inference on the subset and write docs/results.md and figures."""
    from .report import run_all

    out.mkdir(parents=True, exist_ok=True)
    res = run_all(out, stride, frames, backend)
    typer.echo(f"primary variant: {res['primary_variant']}; wrote {out / 'results.md'}")


@app.command()
def render(
    sequence: str,
    out: Path = Path("docs/hero.gif"),
    tiled: bool = True,
    stride: Stride = 3,
    frames: Frames = 70,
    det_thr: float = 0.5,
    width: int = 640,
) -> None:
    """Render an annotated GIF (tracks, trails, density overlay), under 5 MB."""
    from .pipeline import Config, load
    from .render import render_frames, write_gif

    r = load(sequence, Config(tiled=tiled, stride=stride, max_frames=frames, det_thr=det_thr))
    path = write_gif(render_frames(r, width=width), out)
    typer.echo(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


@app.command()
def bench(
    sequence: str = "uav0000086_00000_v",
    n: int = 8,
    backend: str = "frcnn",
) -> None:
    """Time the detector on n frames after warm-up (fresh run, ignores the cache)."""
    import time

    import numpy as np

    from .data import load_sequence
    from .detect import PersonDetector, load_rgb, pick_device
    from .pipeline import DATA_ROOT

    seq = load_sequence(DATA_ROOT, sequence, stride=3, max_frames=n + 1)
    det = PersonDetector(backend, 640)
    imgs = [load_rgb(p) for p in seq.frame_paths]
    det.detect_full(imgs[0])
    tf, tt = [], []
    for im in imgs[1:]:
        t0 = time.perf_counter()
        f = det.detect_full(im)
        t1 = time.perf_counter()
        det.detect_tiled(im, f)
        t2 = time.perf_counter()
        tf.append(t1 - t0)
        tt.append(t2 - t1)
    typer.echo(
        f"device {pick_device()}: full-frame {np.median(tf) * 1000:.0f} ms (median), "
        f"tiled {np.median(tf) * 1000 + np.median(tt) * 1000:.0f} ms"
    )


if __name__ == "__main__":
    app()
