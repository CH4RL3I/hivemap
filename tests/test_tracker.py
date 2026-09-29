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


def test_update_pairs_each_track_with_its_nearest_detection():
    tr = ByteTracker(high_thr=0.5, new_thr=0.5, min_hits=1)
    first = {
        tid: b for tid, b, _ in tr.update(np.array([box(0, 0), box(100, 0)]), np.array([0.9, 0.9]))
    }
    left = min(first, key=lambda t: first[t][0])
    # Detections arrive in the opposite order and each moved by 2 px.
    out = {
        tid: b for tid, b, _ in tr.update(np.array([box(102, 0), box(2, 0)]), np.array([0.9, 0.9]))
    }
    assert len(out) == 2
    centre = lambda b: (b[0] + b[2]) / 2  # noqa: E731
    assert abs(centre(out[left]) - 2) < 5


def test_crossing_tracks_keep_identities():
    """Two people of different size walk through each other; each ID must follow its person."""
    tr = ByteTracker(high_thr=0.5, new_thr=0.5, min_hits=1, max_age=5)
    path: dict[int, list[float]] = {}
    for t in range(41):
        xa, xb = 100 + 5 * t, 300 - 5 * t  # they meet at t = 20
        dets = np.array([box(xa, 200, 20, 40), box(xb, 203, 30, 60)])
        out = tr.update(dets, np.array([0.9, 0.9]))
        assert len(out) == 2
        for tid, b, _ in out:
            path.setdefault(tid, []).append((b[0] + b[2]) / 2)
    assert len(path) == 2  # no identity was created or lost
    for xs in path.values():
        assert len(xs) == 41
        steps = np.abs(np.diff(xs))
        assert steps.max() < 12  # a swap at the crossing would jump by tens of pixels
    slopes = sorted(np.polyfit(range(41), xs, 1)[0] for xs in path.values())
    assert slopes[0] < -4 and slopes[1] > 4  # one ID moves right, the other moves left


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
