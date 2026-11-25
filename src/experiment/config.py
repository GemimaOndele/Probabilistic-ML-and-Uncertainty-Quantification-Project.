"""Strict validation before reserving any experiment output."""
import math


def validate_config(config, *, require_dataset=False):
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    dataset = config.get("dataset")
    if dataset not in (None, "CIFAR-10", "Fashion-MNIST"):
        raise ValueError("dataset must be CIFAR-10, Fashion-MNIST or null")
    if require_dataset and dataset is None:
        raise ValueError("dataset pending team validation (#5)")
    for name in ("split_seed", "mc_passes", "ece_bins"):
        value = config.get(name)
        if type(value) is not int or value < (0 if name == "split_seed" else 1):
            raise ValueError(f"invalid {name}")
    seeds = config.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("training_seeds must contain nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training_seeds must be unique")
    fractions = config.get("train_pool_fractions")
    if not isinstance(fractions, dict) or set(fractions) != {"train", "validation", "calibration"}:
        raise ValueError("train_pool_fractions keys invalid")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in fractions.values()):
        raise ValueError("split fractions must be finite and in (0,1)")
    if not math.isclose(sum(fractions.values()), 1, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    if config.get("preserve_official_test") is not True:
        raise ValueError("official test must be preserved")
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    return config
