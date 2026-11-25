# Compte-rendu — branche `feature/probabilistic-models`

Missions GitHub attribuées à Elias (`eliascre`) : **#12 à #17**. Toutes les expériences ci-dessous ont été exécutées localement ; les chiffres proviennent des fichiers JSON de `results/` (aucun résultat inventé). Matériel : CPU uniquement (`torch 2.15.0+cpu`, `torchvision 0.29.2+cpu`, Python 3.15, Windows).

## État des missions

| Mission | Titre | Fichiers | État |
|---|---|---|---|
| #12 | Veille scientifique Bayesian CNN / MC Dropout | `docs/etat-art-bayesian-cnn.md` | ✅ |
| #13 | Formulation mathématique et probabiliste | `docs/formulation-probabiliste.md` | ✅ |
| #14 | Monte Carlo Dropout | `src/models/cnn.py`, `src/uncertainty/mc_dropout.py`, `scripts/train_baseline.py`, `scripts/run_mc_dropout.py` | ✅ |
| #15 | Incertitude épistémique | `src/uncertainty/mc_dropout.py`, `tests/test_uncertainty.py`, `results/mc_dropout_fashion-mnist_test.json` | ✅ |
| #16 | Visualisation des prédictions incertaines | `scripts/visualiser_incertitude.py`, `results/figures/`, `results/visualisation_incertitude_fashion-mnist.json` | ✅ |
| #17 | Deep Ensembles et coût | `src/uncertainty/decomposition.py`, `scripts/run_ensembles.py`, `results/deep_ensembles_fashion-mnist.json` | ✅ |

## Données

Deux datasets du sujet 5 ont été téléchargés dans `data/raw/` (ignoré par Git) via `scripts/download_datasets.py` :

- **Fashion-MNIST** : 60 000 train / 10 000 test, images 28×28 en niveaux de gris, 10 classes, licence MIT (Xiao et al., 2017).
- **CIFAR-10** : 50 000 train / 10 000 test, images 32×32 couleur, 10 classes (Krizhevsky, 2009).

Le manifeste `results/data_manifest.json` contient les effectifs, classes, forme des images, tailles et empreintes SHA-256 des archives. **Fashion-MNIST est retenu pour les expériences** : coût CPU plus faible que CIFAR-10 et subjectivité visuelle suffisante pour étudier l'incertitude (le benchmark comparatif des datasets reste la mission #5 d'Aminata). Les helpers de splits sont provisoires et doivent être remplacés par les chargeurs officiels de `feature/data-baseline` lors de l'intégration.

Protocole : split train officiel remélangé avec seed 42 en train (50 000) / validation (5 000) / calibration (5 000), test officiel (10 000) inchangé. Les mêmes indices sont utilisés pour le baseline, le MC Dropout et les 3 membres de l'ensemble (les seeds 43 et 44 ne changent que l'initialisation et l'ordre des minibatchs).

## Résultats (Fashion-MNIST, split test, 10 000 images)

### CNN classique (référence déterministe, dropout désactivé)

- Meilleure accuracy de validation : **0,8966** après 3 époques (seed 42, 118,9 s d'entraînement).
- Sur le test : accuracy **0,8913**, NLL **0,2897**.

### MC Dropout (checkpoint seed 42, `T = 30`, dropout `p = 0,25`)

- Accuracy **0,8879**, NLL **0,3036**, temps d'inférence **78,1 s** sur CPU.
- Entropie prédictive moyenne **0,3769 nats**, entropie attendue **0,3283**, information mutuelle **0,0486**.
- MI moyenne sur les prédictions correctes : **0,0416** ; sur les erreurs : **0,1041** (≈ 2,5×). L'incertitude épistémique est donc bien plus élevée là où le modèle se trompe.
- Le passage de `T=1` à `T=30` est nécessaire : avec une seule passe, l'information mutuelle estimée est nulle par construction (test unitaire dédié).

### Deep Ensembles avec dropout (3 membres, seeds 42/43/44, 10 passes MC par membre)

La comparaison est faite **à budget de calcul égal** : 3 membres × 10 passes = **30 passes stochastiques**, comme le MC Dropout `T=30`. Les deux méthodes utilisent le même estimateur (`src/uncertainty/decomposition.py` : entropie de la probabilité moyennée sur toutes les passes moins entropie moyenne des passes). Les valeurs de MI sont donc directement comparables.

- Accuracy **0,8959**, NLL **0,2896**, entropie prédictive **0,3860**, entropie attendue **0,3242**.
- Information mutuelle moyenne **0,0619** (contre **0,0486** pour le MC Dropout au même budget).
- MI correctes **0,0522** ; MI erreurs **0,1451** (≈ 2,8×) — même constat que pour le MC Dropout : l'incertitude épistémique est plus élevée sur les erreurs.
- Diagnostic séparé : désaccord *entre* membres **0,0186** (les membres sont plus d'accord entre eux que ne le sont les masques dropout d'un même réseau).
- Coût : **360,4 s** d'entraînement au total (3 × ~120 s) et **67,7 s** d'inférence, contre 78,1 s pour le MC Dropout à 30 passes. L'ensemble est plus rapide à l'inférence ici, mais coûte ~3× plus cher à l'entraînement.

### Visualisation

`results/figures/exemples_incertitude.png` montre 3 groupes de 5 exemples : erreurs à forte MI, bonnes prédictions à forte MI, erreurs à faible MI. `results/figures/histogramme_mi.png` compare les distributions de MI pour prédictions correctes et erronées. Le rapport JSON donne les indices exacts des exemples, leurs labels, confiances, entropies et MI.

## Commandes de reproduction

```powershell
# 1. Dépendances (deux étapes : index PyTorch pour les wheels CPU, puis PyPI)
python -m pip install torch==2.15.0+cpu torchvision==0.29.2+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-probabilistic-models.txt

# 2. Données
python scripts/download_datasets.py

# 3. Entraînement (seeds indépendantes, même split)
python scripts/train_baseline.py --dataset fashion-mnist --epochs 3 --seed 42
python scripts/train_baseline.py --dataset fashion-mnist --epochs 3 --seed 43 --split-seed 42
python scripts/train_baseline.py --dataset fashion-mnist --epochs 3 --seed 44 --split-seed 42

# 4. Inférence MC Dropout et visualisation
python scripts/run_mc_dropout.py --num-samples 30 --probs results/probs_mc_fashion-mnist_test.npz
python scripts/visualiser_incertitude.py

# 5. Deep Ensembles
python scripts/run_ensembles.py --mc-samples 10

# 6. Tests
python -m pytest tests -q
```

## Limites et points d'intégration

- Le checkpoint baseline est produit localement par notre script : si `feature/data-baseline` fournit un checkpoint officiel plus performant, `scripts/run_mc_dropout.py` accepte un autre `--checkpoint` (le taux de dropout et le jeu de données sont lus dans la config du checkpoint).
- Le temperature scaling, l'ECE, les diagrammes de fiabilité, l'OOD et le conforme relèvent de la mission d'Omar ; nos scripts exportent les probabilités `[T, N, K]` et les tableaux d'incertitude nécessaires, sans réimplémenter ces métriques.
- Aucune expérience CIFAR-10 ni GPU n'a été réalisée ici ; les résultats sont spécifiques au CPU et à Fashion-MNIST.
- Les fichiers binaires (`.pt`, `.npz`) et les indices de splits sont ignorés par Git mais régénérables par les commandes ci-dessus.
