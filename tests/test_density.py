import numpy as np

from hivemap.density import count_errors, density_map


def test_integral_equals_count_interior_and_border():
    pts = np.array([[100, 80], [101, 82], [0, 0], [199, 119], [50, 3]])
    d = density_map(pts, (120, 200), sigma=8)
    assert np.isclose(d.sum(), len(pts), atol=1e-9)
    assert d.min() >= 0


def test_empty_gives_zero_map():
    assert density_map(np.zeros((0, 2)), (30, 40)).sum() == 0


def test_peak_is_at_the_point():
    d = density_map(np.array([[60, 40]]), (100, 100), sigma=5)
    assert np.unravel_index(d.argmax(), d.shape) == (40, 60)


def test_count_errors():
    e = count_errors([3, 5, 10], [4, 5, 8])
    assert np.isclose(e["mae"], 1.0) and np.isclose(e["bias"], 1 / 3)
