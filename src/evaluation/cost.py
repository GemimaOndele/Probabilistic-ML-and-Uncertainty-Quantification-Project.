"""Mesure du coût de calcul (Issue #23)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

import numpy as np


@contextmanager
def timer(store: dict, key: str):
    """``with timer(d, "train_time_s"): ...`` enregistre la durée murale en secondes."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - t0


def benchmark_inference(predict_fn: Callable, inputs, n_repeats: int = 3, warmup: int = 1,
                        n_mc_samples: int = 1, sync: Callable | None = None) -> dict:
    """Temps d'inférence moyen de ``predict_fn(inputs)``.

    ``sync`` (ex. ``torch.cuda.synchronize``) est appelé avant chaque mesure sur GPU.
    ``n_mc_samples`` sert seulement à rapporter un coût par passe stochastique.
    """
    n = len(inputs)
    for _ in range(warmup):
        predict_fn(inputs)
    times = []
    for _ in range(n_repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        predict_fn(inputs)
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
    mean = float(np.mean(times))
    return {
        "inference_time_s": mean,
        "inference_time_std_s": float(np.std(times)),
        "ms_per_image": 1000 * mean / n,
        "ms_per_image_per_pass": 1000 * mean / n / max(n_mc_samples, 1),
        "throughput_img_s": n / mean,
        "n_mc_samples": n_mc_samples,
        "n_images": n,
        "n_repeats": n_repeats,
    }


def count_parameters(model) -> int:
    """Nombre de paramètres entraînables d'un ``torch.nn.Module``."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
