import numpy as np

from hivemap.tracker import ByteTracker, associate, iou_matrix


def box(cx, cy, w=20, h=40):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    assert np.allclose(iou_matrix(a, b), [[1.0, 1 / 3, 0.0]])


def test_associate_picks_best_pairs_and_gates():
    pred = np.array([box(0, 0), box(100, 0)])
    det = np.array([box(101, 0), box(1, 0), box(500, 500)])
    m, un_t, un_d = associate(1 - iou_matrix(pred, det), max_cost=0.7)
    assert sorted(m) == [(0, 1), (1, 0)]
    assert un_t == [] and un_d == [2]


def test_associate_prefers_global_optimum_over_greedy():
    # Greedy would give track 0 its best detection (1) and leave track 1 with a poor one.
    cost = np.array([[0.30, 0.20], [0.25, 0.90]])
    m, _, _ = associate(cost, max_cost=1.0)
    assert sorted(m) == [(0, 1), (1, 0)] or sorted(m) == [(0, 0), (1, 1)]
    total = sum(cost[r, c] for r, c in m)
    assert np.isclose(total, 0.45)  # 0.20 + 0.25 beats 0.30 + 0.90


def test_crossing_tracks_keep_identities():
    """Two people walk through each other along a line; IDs must not swap."""
    tr = ByteTracker(high_thr=0.5, new_thr=0.5, min_hits=1, max_age=5)
    ids_a, ids_b = [], []
    for t in range(41):
        xa, xb = 100 + 5 * t, 300 - 5 * t  # they meet at t = 20
        ya, yb = 200, 200 + (0 if t < 20 else 0)
        # Different sizes so the crossing is resolvable by IoU/motion.
        dets = np.array([box(xa, ya, 20, 40), box(xb, yb + 3, 30, 60)])
        out = tr.update(dets, np.array([0.9, 0.9]))
        by_size = {round(b[3] - b[1]): tid for tid, b, _ in out}
        if 40 in by_size:
            ids_a.append(by_size[40])
        if 60 in by_size:
            ids_b.append(by_size[60])
    assert len(set(ids_a)) == 1 and len(set(ids_b)) == 1
    assert ids_a[0] != ids_b[0]


def test_track_survives_missing_frames_then_recovers_same_id():
    tr = ByteTracker(min_hits=1, max_age=10, new_thr=0.5)
    ids = []
    for t in range(30):
        if 10 <= t < 14:
            out = tr.update(np.zeros((0, 4)), np.zeros(0))
        else:
            out = tr.update(np.array([box(50 + 2 * t, 80)]), np.array([0.9]))
        ids += [i for i, _, _ in out]
    assert len(set(ids)) == 1


def test_low_confidence_detection_extends_track():
    tr = ByteTracker(high_thr=0.5, low_thr=0.1, min_hits=1, max_age=3, new_thr=0.5)
    for t in range(5):
        tr.update(np.array([box(50 + 2 * t, 80)]), np.array([0.9]))
    out = tr.update(np.array([box(60, 80)]), np.array([0.2]))  # weak but well placed
    assert len(out) == 1


def test_spurious_low_confidence_does_not_start_a_track():
    tr = ByteTracker(high_thr=0.5, low_thr=0.1, min_hits=1, new_thr=0.6)
    out = tr.update(np.array([box(50, 80)]), np.array([0.3]))
    assert out == [] and tr.tracks == []
