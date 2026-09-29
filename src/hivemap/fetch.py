"""Download a subset of VisDrone2019-MOT-val without fetching the whole archive.

The official validation archive (about 1.5 GB) is stored uncompressed on Google Drive. Drive
honours HTTP range requests, so we read the zip central directory remotely and pull only the
members belonging to the requested sequences.
"""

from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from pathlib import Path

FILE_ID = "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
ROOT = "VisDrone2019-MOT-val"

# The four validation sequences with the densest pedestrian content (11 to 47 people per frame).
DEFAULT_SEQUENCES = [
    "uav0000086_00000_v",
    "uav0000117_02622_v",
    "uav0000137_00458_v",
    "uav0000339_00001_v",
]


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, with a small block cache."""

    BLOCK = 1 << 20

    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        self._cache: dict[int, bytes] = {}
        head = self._get(0, 0)
        self.size = int(head[1].split("/")[-1])

    def _get(self, start: int, end: int, retries: int = 5) -> tuple[bytes, str]:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read(), r.headers.get("Content-Range", "")
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            start = idx * self.BLOCK
            end = min(start + self.BLOCK, self.size) - 1
            self._cache[idx] = self._get(start, end)[0]
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.BLOCK)
            chunk = self._block(idx)[off : off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch(sequences: list[str] | None, dest: Path, max_frames: int | None = None) -> list[str]:
    """Download annotations and frames for the given sequences into ``dest``."""
    sequences = sequences or DEFAULT_SEQUENCES
    dest.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(RangeFile(URL))
    names = zf.namelist()
    for seq in sequences:
        ann = f"{ROOT}/annotations/{seq}.txt"
        if ann not in names:
            raise SystemExit(f"sequence {seq} not in the validation split")
        (dest / "annotations").mkdir(exist_ok=True)
        (dest / "annotations" / f"{seq}.txt").write_bytes(zf.read(ann))
        frames = sorted(
            n for n in names if n.startswith(f"{ROOT}/sequences/{seq}/") and n.endswith(".jpg")
        )
        if max_frames:
            frames = frames[:max_frames]
        out = dest / "sequences" / seq
        out.mkdir(parents=True, exist_ok=True)
        for i, n in enumerate(frames):
            target = out / Path(n).name
            if not target.exists():
                target.write_bytes(zf.read(n))
            if i % 100 == 0:
                print(f"{seq}: {i}/{len(frames)}", flush=True)
    return sequences


def list_sequences() -> list[str]:
    zf = zipfile.ZipFile(RangeFile(URL))
    pre = f"{ROOT}/annotations/"
    return sorted(Path(n).stem for n in zf.namelist() if n.startswith(pre) and n.endswith(".txt"))
