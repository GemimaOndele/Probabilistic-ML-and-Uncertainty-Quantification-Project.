import numpy as np
import pytest

from src.evaluation import metrics as M


def test_perfect_predictions():
    y = np.array([0, 1, 2, 1])
    p = np.eye(3)[y]
    assert M.accuracy(p, y) == 1.0
    assert M.nll(p, y) == pytest.approx(0.0, abs=1e-9)
    assert M.brier_score(p, y) == pytest.approx(0.0)
    assert M.expected_calibration_error(p, y) == pytest.approx(0.0)


def test_uniform_predictions():
    y = np.array([0, 1, 2, 3])
    p = np.full((4, 4), 0.25)
    assert M.nll(p, y) == pytest.approx(np.log(4))
    assert M.brier_score(p, y) == pytest.approx(0.75 ** 2 + 3 * 0.25 ** 2)


def test_ece_hand_computed():
    # 2 exemples à confiance 0.9 (1 correct), 2 à confiance 0.6 (2 corrects)
    p = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    y = np.array([0, 1, 0, 0])
    # |0.5-0.9|*0.5 + |1.0-0.6|*0.5 = 0.4
    assert M.expected_calibration_error(p, y, n_bins=10) == pytest.approx(0.4)
    assert M.maximum_calibration_error(p, y, n_bins=10) == pytest.approx(0.4)


def test_reliability_bins_counts_sum_to_n():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(10), size=500)
    y = rng.integers(0, 10, 500)
    b = M.reliability_bins(p, y, n_bins=15)
    assert b["counts"].sum() == 500


def test_mutual_information_zero_when_passes_agree():
    p = np.random.default_rng(1).dirichlet(np.ones(5), size=20)
    dec = M.uncertainty_decomposition(np.stack([p] * 8))
    np.testing.assert_allclose(dec["mutual_information"], 0.0, atol=1e-9)
    np.testing.assert_allclose(dec["predictive_entropy"], dec["expected_entropy"])


def test_mutual_information_positive_when_passes_disagree():
    a = np.array([[0.99, 0.01]])
    b = np.array([[0.01, 0.99]])
    dec = M.uncertainty_decomposition(np.stack([a, b]))
    assert dec["predictive_entropy"][0] == pytest.approx(np.log(2))
    assert dec["mutual_information"][0] > 0.6


def test_evaluate_mc_keys_and_shape_errors():
    rng = np.random.default_rng(2)
    mc = rng.dirichlet(np.ones(3), size=(4, 30))
    y = rng.integers(0, 3, 30)
    res = M.evaluate_mc(mc, y)
    for k in ("accuracy", "nll", "brier", "ece", "mean_mutual_information", "n_mc_samples"):
        assert k in res
    assert res["n_mc_samples"] == 4
    with pytest.raises(ValueError):
        M.accuracy(mc, y)
    with pytest.raises(ValueError):
        M.accuracy(mc[0], y[:-1])
