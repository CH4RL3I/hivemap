import numpy as np

from hivemap.kalman import KalmanBoxFilter


def test_constant_velocity_is_learned():
    kf = KalmanBoxFilter(np.array([100.0, 100.0, 20.0, 40.0]))
    for t in range(1, 30):
        kf.predict()
        kf.update(np.array([100.0 + 3 * t, 100.0 - 1 * t, 20.0, 40.0]))
    assert abs(kf.x[4] - 3) < 0.3 and abs(kf.x[5] + 1) < 0.3
    pred = kf.predict()
    assert abs(pred[0] - (100 + 3 * 30)) < 3
    assert abs(pred[1] - (100 - 30)) < 3


def test_update_moves_state_toward_measurement_and_shrinks_covariance():
    kf = KalmanBoxFilter(np.array([50.0, 50.0, 10.0, 20.0]))
    kf.predict()
    trace_before = np.trace(kf.P)
    kf.update(np.array([60.0, 50.0, 10.0, 20.0]))
    assert 50 < kf.x[0] <= 60
    assert np.trace(kf.P) < trace_before


def test_predict_grows_uncertainty():
    kf = KalmanBoxFilter(np.array([0.0, 0.0, 10.0, 20.0]))
    p0 = np.trace(kf.P)
    kf.predict()
    assert np.trace(kf.P) > p0


def test_affine_compensation_translates_state():
    kf = KalmanBoxFilter(np.array([10.0, 20.0, 5.0, 8.0]))
    kf.x[4:6] = [1.0, 2.0]
    kf.apply_affine(np.array([[1.0, 0, 7.0], [0, 1.0, -3.0]]))
    assert np.allclose(kf.x[:2], [17.0, 17.0])
    assert np.allclose(kf.x[4:6], [1.0, 2.0])  # velocity unchanged by pure translation
