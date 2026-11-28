"""Dependency-free validation shared by all prediction producers."""
from math import isfinite


def validate_predictions(sample_ids, labels, probabilities, mc_probabilities=None):
    """Validate [N,K] probabilities and optional [T,N,K] MC samples."""
    n = len(sample_ids)
    if not n or len(set(sample_ids)) != n:
        raise ValueError("sample_ids must be nonempty and unique")
    if len(labels) != n or len(probabilities) != n:
        raise ValueError("prediction lengths differ")
    k = len(probabilities[0])
    if k < 2:
        raise ValueError("at least two classes required")
    def check(rows):
        if len(rows) != n:
            raise ValueError("MC sample length differs")
        for row in rows:
            if len(row) != k or any(not isfinite(p) or p < 0 or p > 1 for p in row):
                raise ValueError("invalid probabilities")
            if abs(sum(row) - 1) > 1e-6:
                raise ValueError("probabilities must sum to one")
    check(probabilities)
    if any(isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < k for y in labels):
        raise ValueError("invalid class labels")
    if mc_probabilities is not None:
        if not len(mc_probabilities):
            raise ValueError("MC passes must be nonempty")
        for rows in mc_probabilities:
            check(rows)
        t = len(mc_probabilities)
        if any(abs(sum(rows[i][j] for rows in mc_probabilities)/t - probabilities[i][j]) > 1e-6
               for i in range(n) for j in range(k)):
            raise ValueError("MC mean differs from probabilities")
