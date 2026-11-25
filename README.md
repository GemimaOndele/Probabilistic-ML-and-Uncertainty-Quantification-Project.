# Probabilistic Machine Learning and Uncertainty Quantification Project

MSc2 IA — ECE Paris 2026–2027. Sujet 5 : classification d'images avec quantification de l'incertitude épistémique.

CNN classique vs MC Dropout ; Deep Ensembles optionnels. CIFAR-10 ou Fashion-MNIST à valider. Mesures : Accuracy, NLL, Brier, ECE, calibration, OOD et coût de calcul.

## Équipe
- Gémima : `feature/project-integration`
- Aminata : `feature/data-baseline`
- Elias : `feature/probabilistic-models`
- Omar : `feature/evaluation-robustness`

PR regroupées vers `dev`, puis vers `main` après revue et tests. Conserver les branches permanentes, utiliser des merge commits, synchroniser depuis dev avant chaque PR. Référencer les Issues dans les commits.

## Livrables
ZIP avec code reproductible et rapport PDF 8–12 pages hors annexes : 5 décembre 2026. Soutenance : 7 décembre.

## Branche `feature/probabilistic-models` (Elias)

Implémentation MC Dropout, quantification de l'incertitude épistémique, visualisation et Deep Ensembles (missions #12 à #17). Le suivi des tickets et la conformité au PDF sont dans [`docs/suivi-missions.md`](docs/suivi-missions.md), le détail des résultats et les commandes de reproduction dans [`docs/compte-rendu-branche.md`](docs/compte-rendu-branche.md), la formulation probabiliste dans [`docs/formulation-probabiliste.md`](docs/formulation-probabiliste.md) et la veille dans [`docs/etat-art-bayesian-cnn.md`](docs/etat-art-bayesian-cnn.md).

Démarrage rapide (CPU) :

```powershell
python -m pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-probabilistic-models.txt
python scripts/download_datasets.py
python scripts/train_baseline.py --dataset fashion-mnist --epochs 3 --seed 42
python scripts/run_mc_dropout.py --num-samples 30
python scripts/visualiser_incertitude.py
python scripts/run_ensembles.py --mc-samples 10
python -m pytest tests -q
```

