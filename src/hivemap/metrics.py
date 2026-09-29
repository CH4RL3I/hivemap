"""Evaluation metrics implemented from their definitions: CLEAR-MOT, IDF1, detection AP.

References: Bernardin and Stiefelhagen 2008 (MOTA); Ristani et al. 2016 (IDF1).
Boxes are xyxy. A match requires IoU >= ``iou_thr`` (0.5 by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker import iou_matrix


@dataclass
class MOTResult:
    mota: float
    idf1: float
    idp: float
    idr: float
    id_switches: int
    fp: int
    fn: int
    num_gt: int
    num_hyp: int
    mostly_tracked: int
    mostly_lost: int
    num_gt_ids: int

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def evaluate_mot(
    gt: list[tuple[np.ndarray, np.ndarray]],
    hyp: list[tuple[np.ndarray, np.ndarray]],
    iou_thr: float = 0.5,
) -> MOTResult:
    """Score a tracking result.

    ``gt`` and ``hyp`` are per-frame lists (same length, same order) of ``(ids, boxes_xyxy)``.
    """
    assert len(gt) == len(hyp)
    fp = fn = idsw = num_gt = num_hyp = 0
    last_match: dict[int, int] = {}  # gt id -> hyp id it was last matched to
    cur: dict[int, int] = {}  # persistent gt id -> hyp id assignment (CLEAR-MOT)
    gt_frames: dict[int, int] = {}
    gt_tracked: dict[int, int] = {}
    # For IDF1: overlap counts between every gt id and every hyp id.
    pair_count: dict[tuple[int, int], int] = {}
    hyp_frames: dict[int, int] = {}

    for (g_ids, g_box), (h_ids, h_box) in zip(gt, hyp, strict=True):
        g_ids = np.asarray(g_ids, dtype=int)
        h_ids = np.asarray(h_ids, dtype=int)
        iou = iou_matrix(g_box, h_box)
        num_gt += len(g_ids)
        num_hyp += len(h_ids)
        for g in g_ids:
            gt_frames[g] = gt_frames.get(g, 0) + 1
        for h in h_ids:
            hyp_frames[h] = hyp_frames.get(h, 0) + 1
        # IDF1 bookkeeping uses every pair above threshold in this frame.
        for i, j in zip(*np.nonzero(iou >= iou_thr), strict=True):
            key = (int(g_ids[i]), int(h_ids[j]))
            pair_count[key] = pair_count.get(key, 0) + 1

        # CLEAR-MOT: first keep previous correspondences that are still valid.
        matched_g: dict[int, int] = {}
        used_h: set[int] = set()
        h_index = {int(h): j for j, h in enumerate(h_ids)}
        for i, g in enumerate(g_ids):
            h = cur.get(int(g))
            if h is not None and h in h_index and iou[i, h_index[h]] >= iou_thr:
                matched_g[i] = h_index[h]
                used_h.add(h_index[h])
        free_g = [i for i in range(len(g_ids)) if i not in matched_g]
        free_h = [j for j in range(len(h_ids)) if j not in used_h]
        if free_g and free_h:
            sub = iou[np.ix_(free_g, free_h)]
            cost = np.where(sub >= iou_thr, 1.0 - sub, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c, strict=True):
                if cost[a, b] < 1e5:
                    matched_g[free_g[a]] = free_h[b]
        for i, j in matched_g.items():
            g, h = int(g_ids[i]), int(h_ids[j])
            if g in last_match and last_match[g] != h:
                idsw += 1
            last_match[g] = h
            cur[g] = h
            gt_tracked[g] = gt_tracked.get(g, 0) + 1
        fn += len(g_ids) - len(matched_g)
        fp += len(h_ids) - len(matched_g)
        for i, g in enumerate(g_ids):
            if i not in matched_g:
                cur.pop(int(g), None)

    mota = 1.0 - (fp + fn + idsw) / max(num_gt, 1)

    # IDF1: best one-to-one assignment of gt trajectories to hyp trajectories.
    gids = sorted(gt_frames)
    hids = sorted(hyp_frames)
    idtp = 0
    if gids and hids:
        gi = {g: k for k, g in enumerate(gids)}
        hi = {h: k for k, h in enumerate(hids)}
        overlap = np.zeros((len(gids), len(hids)))
        for (g, h), n in pair_count.items():
            overlap[gi[g], hi[h]] = n
        r, c = linear_sum_assignment(-overlap)
        idtp = int(overlap[r, c].sum())
    idp = idtp / max(num_hyp, 1)
    idr = idtp / max(num_gt, 1)
    idf1 = 2 * idtp / max(num_gt + num_hyp, 1)

    ratios = [gt_tracked.get(g, 0) / n for g, n in gt_frames.items()]
    return MOTResult(
        mota=mota,
        idf1=idf1,
        idp=idp,
        idr=idr,
        id_switches=idsw,
        fp=fp,
        fn=fn,
        num_gt=num_gt,
        num_hyp=num_hyp,
        mostly_tracked=sum(r >= 0.8 for r in ratios),
        mostly_lost=sum(r <= 0.2 for r in ratios),
        num_gt_ids=len(gids),
    )


def detection_ap(
    gt: list[np.ndarray],
    dets: list[tuple[np.ndarray, np.ndarray]],
    iou_thr: float = 0.5,
    score_thr: float | None = None,
) -> dict[str, float]:
    """Single-class detection AP at ``iou_thr`` (all-point interpolation, VOC 2010 style).

    ``gt``: per-frame xyxy boxes. ``dets``: per-frame ``(boxes, scores)``. When ``score_thr`` is
    given, precision, recall and F1 at that operating point are also returned.
    """
    records: list[tuple[float, int]] = []  # (score, is_tp)
    n_gt = sum(len(g) for g in gt)
    for g, (b, s) in zip(gt, dets, strict=True):
        order = np.argsort(-np.asarray(s))
        b = np.asarray(b).reshape(-1, 4)[order]
        s = np.asarray(s)[order]
        iou = iou_matrix(b, g)
        taken: set[int] = set()
        for k in range(len(b)):
            j = -1
            if iou.shape[1]:
                cand = [(iou[k, jj], jj) for jj in range(iou.shape[1]) if jj not in taken]
                if cand:
                    best, jj = max(cand)
                    if best >= iou_thr:
                        j = jj
            if j >= 0:
                taken.add(j)
            records.append((float(s[k]), int(j >= 0)))
    if not records or n_gt == 0:
        return {"ap": 0.0, "n_gt": n_gt}
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    out = {"ap": float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])), "n_gt": n_gt}
    if score_thr is not None:
        keep = np.array([r[0] >= score_thr for r in records])
        ktp = int(np.sum([r[1] for r, k in zip(records, keep, strict=True) if k]))
        kn = int(keep.sum())
        p = ktp / max(kn, 1)
        r_ = ktp / n_gt
        out.update(precision=p, recall=r_, f1=2 * p * r_ / max(p + r_, 1e-9))
    return out
