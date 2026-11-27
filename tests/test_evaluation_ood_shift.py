import numpy as np
import pytest

from src.evaluation import ood, shift


def test_auroc_perfect_random_and_ties():
    assert ood.auroc([0, 0.1, 0.2], [0.8, 0.9]) == pytest.approx(1.0)
    assert ood.auroc([0.8, 0.9], [0, 0.1]) == pytest.approx(0.0)
    assert ood.auroc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    assert ood.auroc(rng.random(5000), rng.random(5000)) == pytest.approx(0.5, abs=0.03)


def test_aupr_and_fpr95():
    assert ood.aupr([0.0, 0.1], [0.9, 1.0]) == pytest.approx(1.0)
    assert ood.fpr_at_tpr([0.0, 0.1, 0.2, 0.3], [0.9, 1.0]) == pytest.approx(0.0)
    assert ood.fpr_at_tpr([0.95, 0.96], [0.9, 1.0]) == pytest.approx(1.0)


def test_ood_detection_metrics_detects_uncertain_inputs():
    rng = np.random.default_rng(1)
    confident = np.tile(np.eye(5)[rng.integers(0, 5, 200)] * 0.9 + 0.02, (6, 1, 1))
    uncertain = rng.dirichlet(np.ones(5), size=(6, 200))
    res = ood.ood_detection_metrics(confident, uncertain)
    assert res["predictive_entropy"]["auroc"] > 0.95
    assert res["mutual_information"]["auroc"] > 0.95  # passes en désaccord sur OOD


@pytest.mark.parametrize("kind", shift.PERTURBATIONS)
@pytest.mark.parametrize("shape", [(4, 3, 16, 16), (4, 16, 16)])
def test_perturbations_keep_shape_and_range(kind, shape):
    x = np.random.default_rng(0).random(shape)
    for sev in (1, 5):
        out = shift.perturb(x, kind, sev, seed=0)
        assert out.shape == x.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not np.allclose(out, x)


def test_perturbation_is_deterministic_and_validates():
    x = np.random.default_rng(0).random((2, 1, 8, 8))
    np.testing.assert_array_equal(shift.perturb(x, "gaussian_noise", 3, seed=7),
                                  shift.perturb(x, "gaussian_noise", 3, seed=7))
    with pytest.raises(ValueError):
        shift.perturb(x, "inconnue", 1)
    with pytest.raises(ValueError):
        shift.perturb(x, "contrast", 6)


def test_evaluate_under_shift_with_toy_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 64)
    x = np.zeros((64, 1, 8, 8))
    x[y == 1, :, :, :4] = 1.0  # classe 1 : moitié gauche claire

    def predict(imgs):
        s = imgs[:, :, :, :4].mean(axis=(1, 2, 3)) - imgs[:, :, :, 4:].mean(axis=(1, 2, 3))
        p1 = 1 / (1 + np.exp(-10 * (s - 0.5)))
        return np.stack([1 - p1, p1], axis=1)

    rows = shift.evaluate_under_shift(predict, x, y, perturbations=("contrast", "gaussian_blur"),
                                      severities=(1, 5))
    assert rows[0]["perturbation"] == "clean" and rows[0]["accuracy"] == 1.0
    assert len(rows) == 1 + 2 * 2
    contrast5 = [r for r in rows if r["perturbation"] == "contrast" and r["severity"] == 5][0]
    assert contrast5["accuracy_drop"] > 0
    summary = shift.summarize_shift(rows)
    assert set(summary) == {0, 1, 5}
