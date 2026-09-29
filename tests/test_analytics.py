import numpy as np

from hivemap.analytics import analyse, ground_positions
from hivemap.camera import IDENTITY, compose, warp_points
from hivemap.ground import Camera, apply_h


def test_speed_and_flow_recovered_from_synthetic_walkers():
    dt = 1 / 10
    pos = []
    for f in range(60):
        pos.append(
            {
                1: np.array([1.5 * f * dt, 10.0]),  # walks east at 1.5 m/s
                2: np.array([20.0, 30.0 - 1.0 * f * dt]),  # walks south at 1 m/s
            }
        )
    g = analyse(pos, dt, cell=2.0, min_samples=3)
    assert abs(g.track_speeds[1] - 1.5) < 1e-6
    assert abs(g.track_speeds[2] - 1.0) < 1e-6
    ok = np.isfinite(g.flow_u)
    assert np.nanmax(g.flow_u[ok]) > 1.4  # eastward cells exist
    assert np.nanmin(g.flow_v[ok]) < -0.9  # southward cells exist


def test_alert_fires_only_when_crowded():
    dt = 0.1
    sparse = [{i: np.array([i * 15.0, 0.0]) for i in range(4)} for _ in range(5)]
    crowd = [{i: np.array([(i % 5) * 0.4, (i // 5) * 0.4]) for i in range(25)} for _ in range(5)]
    # A sparse frame elsewhere gives the grid a realistic extent.
    g = analyse(sparse + crowd, dt, density_threshold=0.5)
    assert all(f >= 5 for f in g.alert_frames) and len(g.alert_frames) == 5


def test_occupancy_integrates_to_mean_count():
    dt = 0.1
    pos = [{i: np.array([i * 3.0, 5.0]) for i in range(3)} for _ in range(10)]
    g = analyse(pos, dt, cell=2.0)
    cell_area = 2.0 * 2.0
    assert np.isclose(g.occupancy.sum() * cell_area, 3.0, atol=1e-6)


def test_camera_motion_is_compensated_in_ground_positions():
    cam = Camera(1000, 600, f=700.0, pitch=np.radians(50), cam_height=20.0)
    foot = np.array([[500.0, 500.0]])
    shift = np.array([[1.0, 0, 30.0], [0, 1.0, 12.0]])  # frame 1 content moved by (30, 12) px
    to_ref = [IDENTITY.copy(), compose(IDENTITY, shift)]
    moved = warp_points(shift, foot)[0]
    box = lambda p: np.array([p[0] - 5, p[1] - 30, p[0] + 5, p[1]])  # noqa: E731
    pos = ground_positions([[(1, box(foot[0]))], [(1, box(moved))]], cam, to_ref)
    assert np.allclose(pos[0][1], pos[1][1], atol=1e-6)
    assert np.allclose(pos[0][1], apply_h(cam.homography(), foot)[0])
