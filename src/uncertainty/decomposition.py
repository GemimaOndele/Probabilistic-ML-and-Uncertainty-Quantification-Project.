"""Décomposition générique entropie / information mutuelle.

Accepte les probabilités MC d'un seul réseau ``[T, N, K]`` ou celles d'un
ensemble de réseaux avec dropout ``[M, T, N, K]``. Dans les deux cas, on
moyenne sur **toutes les passes stochastiques** avant de calculer :

    H_préd = H(mean_s p_s)
    H_attendue = mean_s H(p_s)
    MI = H_préd - H_attendue

Cela rend l'information mutuelle du MC Dropout (M=1) et celle du Deep
Ensembles avec dropout (M>1) directement comparables, à budget de passes
égal. Un diagnostic supplémentaire mesure le désaccord *entre* membres
après moyennage de leurs passes internes.
"""

from __future__ import annotations

import numpy as np

from src.uncertainty.mc_dropout import entropy


def decompose_uncertainty(probs: np.ndarray) -> dict:
    """Décompose l'incertitude pour ``[T,N,K]`` ou ``[M,T,N,K]``."""
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim not in (3, 4):
        raise ValueError(f"probs doit etre [T,N,K] ou [M,T,N,K], recu {p.shape}")
    if p.ndim == 3:
        flat = p
        num_members = 1
    else:
        num_members, passes, n, k = p.shape
        flat = p.reshape(num_members * passes, n, k)

    total_passes, n, k = flat.shape
    if total_passes < 1 or n < 1 or k < 2:
        raise ValueError(f"dimensions invalides : {p.shape}")

    p_mean = flat.mean(axis=0)
    predictive_entropy = entropy(p_mean)
    expected_entropy = entropy(flat).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)

    result = {
        "num_members": num_members,
        "num_passes_per_member": total_passes // num_members,
        "total_passes": total_passes,
        "num_examples": n,
        "num_classes": k,
        "probs_mean": p_mean,
        "predicted_labels": p_mean.argmax(axis=1),
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    }

    # Diagnostic propre à l'ensemble : désaccord entre membres après
    # moyennage interne de leurs passes dropout.
    if p.ndim == 4:
        per_member_mean = p.mean(axis=1)  # [M, N, K]
        ensemble_mean = per_member_mean.mean(axis=0)
        between_entropy = entropy(ensemble_mean) - entropy(per_member_mean).mean(axis=0)
        result["between_member_mi"] = np.maximum(between_entropy, 0.0)
        result["per_member_mean_probs"] = per_member_mean
    return result
