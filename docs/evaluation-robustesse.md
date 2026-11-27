# Évaluation, calibration et robustesse (`src/evaluation`)

Responsable : Omar, branche `feature/evaluation-robustness`. Module en NumPy pur
(matplotlib seulement pour les figures) : il accepte des tableaux NumPy ou des
tenseurs PyTorch, sans dépendre de PyTorch.

## Entrées (contrat technique)
- CNN classique : logits `[N, C]` ou probabilités `[N, C]`.
- MC Dropout / Deep Ensembles : probabilités `[T, N, C]` (une passe ou un membre par T).
- Étiquettes entières `[N]`. Images pour les perturbations : `[N, C, H, W]` ou `[N, H, W]` dans [0, 1], avant normalisation.

## Contenu
| Fichier | Rôle | Issue |
|---|---|---|
| `metrics.py` | Accuracy, NLL, Brier, ECE, MCE, bins de fiabilité, entropie prédictive, entropie attendue, information mutuelle | #19 |
| `calibration.py` | Temperature Scaling ajusté sur le split calibration seul ; diagrammes de fiabilité | #20 |
| `ood.py` | Scores d'incertitude, AUROC, AUPR, FPR@95%TPR | #21 |
| `shift.py` | 7 perturbations × 5 sévérités (bruit, sel-poivre, flou, contraste, luminosité, occlusion, rotation) et évaluation de la dégradation | #21 |
| `conformal.py` | Prédiction conforme split (LAC, APS), couverture, taille des ensembles, couverture par classe | #22 |
| `cost.py`, `report.py` | Temps d'entraînement et d'inférence, export JSON/CSV (méthode, dataset, split, seed), tableau comparatif | #23 |
| `evaluate.py` | Pipeline complet à partir de prédictions `.npz` sauvegardées | #19 à #23 |

## Utilisation
Sauvegarder les prédictions de chaque méthode par split dans un `.npz` contenant `labels`
et l'une des clés `logits`, `probs` ou `mc_probs`, puis :

```bash
python -m src.evaluation.evaluate --method mc_dropout --dataset cifar10 --seed 42 \
    --cal results/predictions/mc_dropout_calibration.npz \
    --test results/predictions/mc_dropout_test.npz \
    --ood results/predictions/mc_dropout_ood.npz \
    --out results/evaluation --plots
```

Robustesse aux perturbations, avec n'importe quelle fonction de prédiction :

```python
from src.evaluation import evaluate_under_shift, summarize_shift, save_csv
rows = evaluate_under_shift(predict_fn, x_test, y_test, seed=42)  # predict_fn -> [N,C] ou [T,N,C]
save_csv([{"method": "mc_dropout", "dataset": "cifar10", "split": "test", "seed": 42, **r} for r in rows],
         "results/evaluation/mc_dropout_shift.csv")
```

## Choix méthodologiques
- ECE à 15 bins de largeur égale (Guo et al., 2017) ; MCE rapportée en complément.
- Brier multiclasse (somme sur les classes, entre 0 et 2).
- Information mutuelle = entropie prédictive − entropie attendue (Gal, 2016 ; Depeweg et al., 2018) : nulle pour un modèle déterministe.
- Temperature Scaling : NLL minimisée sur log T par section dorée (NLL convexe en 1/T). Pour MC Dropout, chaque passe est remise à l'échelle avant la moyenne.
- OOD : les exemples hors distribution sont la classe positive ; score = 1 − max p, entropie ou information mutuelle.
- Conforme : quantile ⌈(n+1)(1−α)⌉/n sur le split calibration, appliqué au test ; le même split calibration que le Temperature Scaling.

## Références
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural Networks*. ICML.
- Gal, Ghahramani (2016). *Dropout as a Bayesian Approximation*. ICML.
- Depeweg et al. (2018). *Decomposition of Uncertainty in Bayesian Deep Learning*. ICML.
- Hendrycks, Gimpel (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples*. ICLR.
- Hendrycks, Dietterich (2019). *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*. ICLR.
- Ovadia et al. (2019). *Can You Trust Your Model's Uncertainty?* NeurIPS.
- Sadinle, Lei, Wasserman (2019). *Least Ambiguous Set-Valued Classifiers*. JASA.
- Romano, Sesia, Candès (2020). *Classification with Valid and Adaptive Coverage*. NeurIPS.
- Angelopoulos, Bates (2021). *A Gentle Introduction to Conformal Prediction*.
