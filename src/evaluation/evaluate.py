"""Pipeline d'évaluation à partir de prédictions sauvegardées.

Format d'entrée (``.npz``), un fichier par split et par méthode :
- ``labels`` : ``[N]`` entiers (absent autorisé pour un jeu OOD) ;
- et **un** des tableaux : ``logits`` ``[N, C]``, ``probs`` ``[N, C]`` ou
  ``mc_probs`` ``[T, N, C]``.

Exemple ::

    python -m src.evaluation.evaluate --method mc_dropout --dataset cifar10 --seed 42 \
        --cal results/predictions/mc_dropout_calibration.npz \
        --test results/predictions/mc_dropout_test.npz \
        --ood results/predictions/mc_dropout_svhn.npz \
        --out results/evaluation --plots

Produit ``<method>_metrics.json``, ``<method>_metrics.csv`` et, avec ``--plots``,
les diagrammes de fiabilité avant/après Temperature Scaling.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._utils import softmax
from .calibration import TemperatureScaler, plot_reliability_diagram
from .conformal import SplitConformalClassifier
from .metrics import evaluate_mc
from .ood import ood_detection_metrics
from .report import comparison_table, make_record, save_csv, save_json


def load_predictions(path) -> dict:
    """Charge un ``.npz`` et renvoie ``{"mc_probs": [T,N,C], "labels", "logits"?}``."""
    d = np.load(path)
    out = {"labels": d["labels"] if "labels" in d else None}
    if "mc_probs" in d:
        mc = d["mc_probs"]
        out["mc_probs"] = mc if mc.ndim == 3 else mc[None]
    elif "probs" in d:
        out["mc_probs"] = d["probs"][None]
    elif "logits" in d:
        out["logits"] = d["logits"]
        out["mc_probs"] = softmax(d["logits"])[None]
    else:
        raise KeyError(f"{path} : clé 'mc_probs', 'probs' ou 'logits' attendue")
    return out


def _fit_scaler(cal: dict) -> tuple[TemperatureScaler, bool]:
    if "logits" in cal:
        return TemperatureScaler().fit(cal["logits"], cal["labels"]), False
    return TemperatureScaler().fit(cal["mc_probs"], cal["labels"], from_probs=True), True


def run(method: str, dataset: str, seed: int, cal_path, test_path, ood_path=None,
        out_dir="results/evaluation", alpha: float = 0.1, n_bins: int = 15,
        plots: bool = False) -> dict:
    cal, test = load_predictions(cal_path), load_predictions(test_path)
    out_dir = Path(out_dir)
    common = {"method": method, "dataset": dataset, "seed": seed}

    scaler, from_probs = _fit_scaler(cal)
    test_scaled_mc = (scaler.transform_mc(test["mc_probs"]) if from_probs
                      else softmax(test["logits"] / scaler.temperature)[None])

    raw = evaluate_mc(test["mc_probs"], test["labels"], n_bins)
    ts = evaluate_mc(test_scaled_mc, test["labels"], n_bins)
    records = [
        make_record(method, dataset, "test", seed, raw, calibration="none"),
        make_record(f"{method}+TS", dataset, "test", seed, ts, calibration="temperature",
                    temperature=scaler.temperature),
    ]
    result = {**common, "temperature_scaling": scaler.to_dict(),
              "test": raw, "test_temperature_scaled": ts}

    cal_scaled = (scaler.transform(cal["mc_probs"]) if from_probs
                  else scaler.transform(cal["logits"]))
    test_scaled = test_scaled_mc.mean(axis=0)
    result["conformal"] = {
        m: SplitConformalClassifier(alpha=alpha, method=m, seed=seed)
        .fit(cal_scaled, cal["labels"]).evaluate(test_scaled, test["labels"])
        for m in ("lac", "aps")
    }

    if ood_path is not None:
        ood = load_predictions(ood_path)
        result["ood"] = ood_detection_metrics(test["mc_probs"], ood["mc_probs"])

    if plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability_diagram(test["mc_probs"].mean(0), test["labels"], n_bins,
                                 title=f"{method}", path=out_dir / f"{method}_reliability.png")
        plot_reliability_diagram(test_scaled, test["labels"], n_bins, title=f"{method} + TS",
                                 path=out_dir / f"{method}_reliability_ts.png")

    save_json(result, out_dir / f"{method}_metrics.json")
    save_csv(records, out_dir / f"{method}_metrics.csv")
    result["table"] = comparison_table(records)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cal", required=True, help="prédictions du split calibration (.npz)")
    ap.add_argument("--test", required=True, help="prédictions du split test (.npz)")
    ap.add_argument("--ood", help="prédictions sur un jeu hors distribution (.npz)")
    ap.add_argument("--out", default="results/evaluation")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.method, a.dataset, a.seed, a.cal, a.test, a.ood, a.out, a.alpha, a.n_bins,
              a.plots)
    print(res["table"])


if __name__ == "__main__":
    main()
