"""Export des résultats comparables en JSON / CSV et tableau comparatif (Issue #23).

Chaque enregistrement contient au minimum : méthode, dataset, split, seed et
métriques, conformément au contrat technique. Aucun résultat n'est inventé :
ces fonctions n'écrivent que ce qu'on leur donne.
"""
from __future__ import annotations

import csv
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REQUIRED = ("method", "dataset", "split", "seed")
TABLE_COLUMNS = ("method", "split", "accuracy", "nll", "brier", "ece",
                 "mean_mutual_information", "temperature", "inference_time_s",
                 "train_time_s")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def make_record(method: str, dataset: str, split: str, seed: int, metrics: dict,
                **extra) -> dict:
    rec = {"method": method, "dataset": dataset, "split": split, "seed": int(seed),
           **{k: v for k, v in metrics.items() if np.isscalar(v) or v is None}, **extra}
    rec["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rec["python"] = platform.python_version()
    return rec


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_csv(records: list[dict], path) -> Path:
    """CSV avec l'union des clés scalaires ; colonnes obligatoires en premier."""
    for r in records:
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            raise ValueError(f"champs obligatoires manquants : {missing}")
    keys = list(REQUIRED)
    for r in records:
        keys += [k for k, v in r.items()
                 if k not in keys and (np.isscalar(v) or v is None)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: _jsonable(r.get(k)) for k in keys})
    return path


def comparison_table(records: list[dict], columns=TABLE_COLUMNS, fmt: str = "{:.4f}") -> str:
    """Tableau Markdown à protocole constant (une ligne par méthode × split)."""
    cols = [c for c in columns if any(c in r for r in records)]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in records:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(fmt.format(v) if isinstance(v, float) and math.isfinite(v)
                         else ("" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
