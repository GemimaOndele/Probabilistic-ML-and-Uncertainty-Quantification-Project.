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

Les commandes d'exécution seront complétées après implémentation.


## Installation et vérification
Python 3.11 (CI de référence), CPU. Depuis la racine :
```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest
python scripts/check_environment.py
python -m pip freeze > results/environment.txt
```
Les dépendances directes sont figées ; environment.txt enregistre la résolution transitive de la machine. Le fichier requirements.txt ne constitue pas un verrou multi-plateforme complet. GPU : environnement séparé à valider et documenter avant comparaison.

Documents : [cadrage](docs/cadrage.md), [architecture](docs/architecture.md), [interfaces](docs/contrat-technique.md), [protocole](docs/protocole-experimental.md), [contrôle GitHub](docs/validation-github.md).

La commande de smoke test valide l'environnement et des probabilités synthétiques ; les entraînements et résultats scientifiques restent à implémenter dans les tickets spécialisés. Dataset non validé : configs/experiment.json conserve dataset=null.
Pour les expériences futures, fixer aussi NumPy, PyTorch, CUDA et les workers de chargement ; consigner les limites de déterminisme matériel et les versions. PYTHONHASHSEED doit être défini avant le lancement si le code dépend du hash Python.
