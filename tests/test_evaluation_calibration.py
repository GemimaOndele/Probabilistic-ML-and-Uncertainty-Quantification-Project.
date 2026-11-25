import numpy as np
import pytest

from src.evaluation._utils import softmax
from src.evaluation.calibration import TemperatureScaler, plot_reliability_diagram
from src.evaluation.metrics import expected_calibration_error, nll


def _make_overconfident(n=4000, c=10, true_t=3.0, seed=0):
    """Logits « vrais » z ; le modèle sur-confiant renvoie z * true_t."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, size=(n, c))
    y = np.array([rng.choice(c, p=row) for row in softmax(z)])
    return z * true_t, y


def test_temperature_recovers_known_value():
    logits, y = _make_overconfident()
    cal, test = slice(0, 2000), slice(2000, None)
    ts = TemperatureScaler().fit(logits[cal], y[cal])
    assert ts.temperature == pytest.approx(3.0, rel=0.1)
    before = softmax(logits[test])
    after = ts.transform(logits[test])
    assert nll(after, y[test]) < nll(before, y[test])
    assert expected_calibration_error(after, y[test]) < expected_calibration_error(before, y[test])
    assert ts.nll_after <= ts.nll_before


def test_temperature_on_mc_probs():
    logits, y = _make_overconfident(n=2000, seed=1)
    rng = np.random.default_rng(3)
    mc = np.stack([softmax(logits + rng.normal(0, 0.3, logits.shape)) for _ in range(5)])
    ts = TemperatureScaler().fit(mc, y, from_probs=True)
    assert ts.temperature > 1.5
    assert ts.transform(mc).shape == (2000, 10)
    assert ts.transform_mc(mc).shape == mc.shape


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TemperatureScaler().transform(np.zeros((2, 3)))


def test_reliability_diagram_saves(tmp_path):
    pytest.importorskip("matplotlib")
    logits, y = _make_overconfident(n=300)
    path = tmp_path / "rel.png"
    plot_reliability_diagram(softmax(logits), y, path=str(path), title="test")
    assert path.exists() and path.stat().st_size > 0
