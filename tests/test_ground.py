import numpy as np

from hivemap.ground import Camera, apply_h, fit_camera, predict_heights


def test_homography_round_trip():
    cam = Camera(1344, 756, f=800.0, pitch=np.radians(55), cam_height=25.0)
    H = cam.homography()
    rng = np.random.default_rng(0)
    # Points below the horizon only (rays that hit the ground).
    pts = np.c_[rng.uniform(0, 1344, 200), rng.uniform(400, 756, 200)]
    ground = apply_h(H, pts)
    back = apply_h(np.linalg.inv(H), ground)
    assert np.allclose(back, pts, atol=1e-6)


def test_centre_pixel_hits_expected_ground_point():
    cam = Camera(1000, 600, f=700.0, pitch=np.radians(45), cam_height=20.0)
    X, Z = apply_h(cam.homography(), np.array([[500.0, 300.0]]))[0]
    assert abs(X) < 1e-9
    assert np.isclose(Z, 20.0 / np.tan(np.radians(45)))  # looking 45 deg down from 20 m


def test_predict_heights_matches_pinhole_projection():
    cam = Camera(1344, 756, f=800.0, pitch=np.radians(50), cam_height=30.0)
    H = cam.homography()
    feet = apply_h(np.linalg.inv(H), np.array([[5.0, 30.0], [-10.0, 50.0]]))
    h = predict_heights(cam, feet)
    assert np.all(h > 0)
    assert h[0] > h[1]  # nearer people look taller


def test_fit_camera_recovers_synthetic_geometry():
    """Synthesise 1.7 m people seen by a known camera and recover its height and pitch."""
    f = 0.5 * 1344 / np.tan(np.radians(40))
    true = Camera(1344, 756, f=f, pitch=np.radians(35), cam_height=30.0)
    Hinv = np.linalg.inv(true.homography())
    rng = np.random.default_rng(1)
    boxes = []
    while len(boxes) < 400:
        g = np.array([[rng.uniform(-30, 30), rng.uniform(25, 90)]])
        foot = apply_h(Hinv, g)[0]
        if not (0 < foot[1] < 756 and 0 < foot[0] < 1344):
            continue
        h = predict_heights(true, foot[None])[0] * np.exp(rng.normal(0, 0.05))
        boxes.append([foot[0] - h / 6, foot[1] - h, foot[0] + h / 6, foot[1]])
    cam, _ = fit_camera(np.array(boxes), 1344, 756, hfov_deg=80.0)
    assert abs(cam.cam_height - 30.0) / 30.0 < 0.2
    assert abs(np.degrees(cam.pitch) - 35) < 8
