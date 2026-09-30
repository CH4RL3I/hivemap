from pathlib import Path

import numpy as np

from hivemap.data import Sequence
from hivemap.pipeline import Config, Run
from hivemap.report import recall_by_height, threshold_sweep


def tiny_run():
    # One frame, two GT persons (heights 30 and 90 px); the detector finds only the tall one.
    gt = {1: np.array([[1, 10, 10, 10, 30], [2, 100, 10, 30, 90]], dtype=float)}
    seq = Sequence("toy", [Path("0000001.jpg")], gt, {1: np.zeros((0, 4))})
    dets = [(np.array([[100, 10, 130, 100]], dtype=np.float32), np.array([0.9], dtype=np.float32))]
    return Run(seq, Config(det_thr=0.5), dets, np.zeros(1), [np.eye(2, 3)], np.zeros(1))


def test_recall_by_height_counts_found_persons_per_bin():
    r = recall_by_height([tiny_run()])
    assert r["0-40 px"] == (0, 1)
    assert r["70-110 px"] == (1, 1)


def test_threshold_sweep_runs_on_toy_data():
    s = threshold_sweep([tiny_run()], grid=(0.5,))
    assert s["0.5"]["recall"] == 0.5 and s["0.5"]["precision"] == 1.0
