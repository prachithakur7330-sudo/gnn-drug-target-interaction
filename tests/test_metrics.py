import numpy as np

from dti.metrics import concordance_index, mse, pearson, r_squared, rmse, rm2


def test_perfect_prediction():
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    assert mse(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert np.isclose(pearson(y, y), 1.0)
    assert np.isclose(concordance_index(y, y), 1.0)
    assert np.isclose(r_squared(y, y), 1.0)


def test_concordance_index_known_value():
    # one pair out of order -> CI = 5/6
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 4.0, 3.0])
    assert np.isclose(concordance_index(y, p), 5 / 6)


def test_concordance_index_reversed_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(concordance_index(y, -y), 0.0)


def test_r_squared_can_go_negative():
    y = np.array([5.0, 6.0, 7.0])
    p = np.array([7.0, 6.0, 5.0])      # worse than predicting the mean
    assert r_squared(y, p) < 0


def test_mse_matches_numpy():
    rng = np.random.default_rng(0)
    y, p = rng.normal(size=50), rng.normal(size=50)
    assert np.isclose(mse(y, p), np.mean((y - p) ** 2))


def test_rm2_between_zero_and_r2():
    rng = np.random.default_rng(1)
    y = rng.normal(size=100)
    p = y + rng.normal(scale=0.3, size=100)
    assert 0 < rm2(y, p) <= pearson(y, p) ** 2 + 1e-9
