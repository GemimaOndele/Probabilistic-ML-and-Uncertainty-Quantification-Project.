import json

import numpy as np
import pytest

from src.evaluation import cost, report
from src.evaluation._utils import softmax
from src.evaluation.conformal import SplitConformalClassifier
from src.evaluation.evaluate import run


def _data(n, c=10, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, size=(n, c))
    p = softmax(z)
    y = np.array([rng.choice(c, p=row) for row in p])
    return p, y


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_conformal_coverage_guarantee(method):
    p_cal, y_cal = _data(3000, seed=0)
    p_test, y_test = _data(5000, seed=1)
    cp = SplitConformalClassifier(alpha=0.1, method=method).fit(p_cal, y_cal)
    res = cp.evaluate(p_test, y_test)
    assert res["coverage"] >= 0.9 - 0.02
    assert 1.0 <= res["mean_set_size"] <= 10.0 or method == "aps"
    assert len(res["per_class_coverage"]) == 10


def test_conformal_smaller_alpha_gives_larger_sets():
    p_cal, y_cal = _data(2000)
    p_test, _ = _data(1000, seed=2)
    s01 = SplitConformalClassifier(alpha=0.01).fit(p_cal, y_cal).predict_sets(p_test).sum(1)
    s20 = SplitConformalClassifier(alpha=0.2).fit(p_cal, y_cal).predict_sets(p_test).sum(1)
    assert s01.mean() > s20.mean()


def test_report_csv_json_and_table(tmp_path):
    rec = report.make_record("baseline", "cifar10", "test", 42,
                             {"accuracy": 0.9, "nll": 0.3, "ece": float("nan"),
                              "per_class": [1, 2]})
    assert "per_class" not in rec
    csv_path = report.save_csv([rec], tmp_path / "m.csv")
    header = csv_path.read_text().splitlines()[0].split(",")
    assert header[:4] == ["method", "dataset", "split", "seed"]
    js = report.save_json({"x": np.float32(1.5), "nan": float("nan"), "a": np.arange(3)},
                          tmp_path / "m.json")
    assert json.loads(js.read_text()) == {"x": 1.5, "nan": None, "a": [0, 1, 2]}
    assert "| method |" in report.comparison_table([rec])
    with pytest.raises(ValueError):
        report.save_csv([{"method": "x"}], tmp_path / "bad.csv")


def test_cost_helpers():
    d = {}
    with cost.timer(d, "t"):
        sum(range(1000))
    assert d["t"] >= 0
    b = cost.benchmark_inference(lambda x: x * 2, np.ones(100), n_repeats=2, n_mc_samples=10)
    assert b["n_images"] == 100 and b["ms_per_image_per_pass"] <= b["ms_per_image"]


def test_end_to_end_pipeline(tmp_path):
    rng = np.random.default_rng(0)
    c = 5
    def split(n, seed):
        r = np.random.default_rng(seed)
        z = r.normal(0, 1.5, (n, c))
        y = np.array([r.choice(c, p=row) for row in softmax(z)])
        mc = np.stack([softmax(3 * z + r.normal(0, 0.5, z.shape)) for _ in range(4)])
        return mc, y
    mc_cal, y_cal = split(800, 1)
    mc_test, y_test = split(800, 2)
    mc_ood = rng.dirichlet(np.ones(c), size=(4, 300))
    np.savez(tmp_path / "cal.npz", mc_probs=mc_cal, labels=y_cal)
    np.savez(tmp_path / "test.npz", mc_probs=mc_test, labels=y_test)
    np.savez(tmp_path / "ood.npz", mc_probs=mc_ood)
    pytest.importorskip("matplotlib")
    res = run("mc_dropout", "toy", 0, tmp_path / "cal.npz", tmp_path / "test.npz",
              tmp_path / "ood.npz", tmp_path / "out", plots=True)
    assert res["temperature_scaling"]["temperature"] > 1.5
    assert res["test_temperature_scaled"]["nll"] < res["test"]["nll"]
    assert {"lac", "aps"} == set(res["conformal"])
    assert "mutual_information" in res["ood"]
    for f in ("mc_dropout_metrics.json", "mc_dropout_metrics.csv",
              "mc_dropout_reliability.png", "mc_dropout_reliability_ts.png"):
        assert (tmp_path / "out" / f).exists()
