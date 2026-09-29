import numpy as np

from hivemap.data import in_ignore, load_sequence, parse_annotations, xywh_to_xyxy


def write_ann(tmp_path):
    # frame, id, x, y, w, h, score, category, truncation, occlusion
    rows = [
        "1,1,10,10,20,40,1,1,0,0",  # pedestrian
        "1,2,100,10,20,40,1,2,0,0",  # people
        "1,3,200,10,50,30,1,4,0,0",  # car: not a person
        "1,4,300,10,20,40,0,1,0,0",  # score 0: not counted
        "1,-1,500,500,100,100,0,0,0,0",  # ignored region (category 0)
        "2,1,12,10,20,40,1,1,0,0",
    ]
    p = tmp_path / "seq.txt"
    p.write_text("\n".join(rows) + "\n")
    return p


def test_parse_keeps_only_scored_persons(tmp_path):
    gt, ign = parse_annotations(write_ann(tmp_path))
    assert sorted(gt) == [1, 2]
    assert sorted(gt[1][:, 0]) == [1, 2]
    assert ign[1].shape == (1, 4)


def test_in_ignore_uses_area_fraction():
    ign = np.array([[0, 0, 100, 100]])
    boxes = np.array([[10, 10, 30, 30], [90, 0, 130, 20], [200, 200, 220, 220]], dtype=float)
    assert in_ignore(boxes, ign).tolist() == [True, False, False]


def test_box_conversion_round_trip():
    b = np.array([[5.0, 6.0, 20.0, 30.0]])
    assert np.allclose(xywh_to_xyxy(b), [[5, 6, 25, 36]])


def test_load_sequence_only_uses_annotated_frames(tmp_path):
    (tmp_path / "annotations").mkdir()
    ann = write_ann(tmp_path)
    ann.rename(tmp_path / "annotations" / "s.txt")
    d = tmp_path / "sequences" / "s"
    d.mkdir(parents=True)
    for i in (1, 2, 3):
        (d / f"{i:07d}.jpg").write_bytes(b"x")
    seq = load_sequence(tmp_path, "s")
    assert seq.frame_numbers() == [1, 2]
