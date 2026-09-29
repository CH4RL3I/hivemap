import numpy as np

from hivemap.metrics import detection_ap, evaluate_mot


def frame(ids, boxes):
    return np.array(ids, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4)


A = [0, 0, 10, 10]
B = [50, 50, 60, 60]


def test_perfect_tracking():
    gt = [frame([1, 2], [A, B])] * 4
    r = evaluate_mot(gt, gt)
    assert r.mota == 1.0 and r.idf1 == 1.0 and r.id_switches == 0


def test_known_toy_example():
    """4 frames, 1 GT person. Hyp: id 7 for frames 1-2, a miss in 3, id 9 in 4, plus one FP."""
    gt = [frame([1], [A])] * 4
    hyp = [
        frame([7], [A]),
        frame([7], [A]),
        frame([], []),
        frame([9, 5], [A, B]),
    ]
    r = evaluate_mot(gt, hyp)
    assert (r.fn, r.fp, r.id_switches, r.num_gt) == (1, 1, 1, 4)
    assert np.isclose(r.mota, 1 - (1 + 1 + 1) / 4)
    # Best ID assignment: gt1 <-> hyp7 gives IDTP = 2; num_hyp = 4; IDF1 = 2*2/(4+4)
    assert np.isclose(r.idf1, 0.5)


def test_id_switch_counted_once_for_a_persistent_change():
    gt = [frame([1], [A])] * 6
    hyp = [frame([3], [A])] * 3 + [frame([4], [A])] * 3
    r = evaluate_mot(gt, hyp)
    assert r.id_switches == 1 and r.fp == 0 and r.fn == 0
    assert np.isclose(r.idf1, 0.5)  # IDTP = 3, IDF1 = 2*3/(6+6)


def test_iou_threshold_respected():
    gt = [frame([1], [[0, 0, 10, 10]])]
    hyp = [frame([1], [[6, 0, 16, 10]])]  # IoU = 4/16 = 0.25
    r = evaluate_mot(gt, hyp)
    assert r.fn == 1 and r.fp == 1


def test_detection_ap_perfect_and_half():
    gt = [np.array([A, B], dtype=float)]
    perfect = [(np.array([A, B], dtype=float), np.array([0.9, 0.8]))]
    assert np.isclose(detection_ap(gt, perfect)["ap"], 1.0)
    # One correct detection ranked first, the second GT never found: AP = 0.5.
    half = [(np.array([A], dtype=float), np.array([0.9]))]
    assert np.isclose(detection_ap(gt, half)["ap"], 0.5)


def test_detection_ap_penalises_confident_false_positive():
    gt = [np.array([A], dtype=float)]
    d = [(np.array([[200, 200, 210, 210], A], dtype=float), np.array([0.99, 0.5]))]
    r = detection_ap(gt, d, score_thr=0.4)
    assert np.isclose(r["ap"], 0.5)  # precision at recall 1 is 1/2
    assert np.isclose(r["recall"], 1.0) and np.isclose(r["precision"], 0.5)
