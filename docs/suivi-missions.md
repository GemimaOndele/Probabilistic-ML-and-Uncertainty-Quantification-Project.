# Suivi des tickets #12 → #17 et conformité au PDF du module

Ce document fait le lien entre le travail attendu sur les tickets du GitHub Project (assignés à `eliascre`, branche `feature/probabilistic-models`), les preuves produites localement, et les exigences méthodologiques obligatoires du PDF « Projet de groupe » (sujet 5).

## 1. Tickets attribués à Elias

| Ticket | Travail attendu (texte du ticket) | Preuve dans la branche | État |
|---|---|---|---|
| **#12 — [P0] Veille scientifique : Bayesian CNN et MC Dropout** | « Références, hypothèses et limites de l'approximation bayésienne. » | `docs/etat-art-bayesian-cnn.md` (9 références datées, 4 hypothèses, 7 limites) | ✅ fait |
| **#13 — [P0] Formulation mathématique et probabiliste** | « Vraisemblance catégorielle, distribution prédictive et approximation variationnelle. » | `docs/formulation-probabiliste.md` (vraisemblance, ELBO, prédictive MC, décomposition entropie/MI) | ✅ fait |
| **#14 — [P0] Implémentation du Monte Carlo Dropout** | « Réutiliser checkpoint baseline, activer dropout à l'inférence et garder batchnorm en eval. » | `src/models/cnn.py` (`mc_dropout_mode`), `scripts/train_baseline.py`, `scripts/run_mc_dropout.py`, checkpoint `seed42`, `results/mc_dropout_fashion-mnist_test.json` | ✅ fait |
| **#15 — [P0] Quantification de l'incertitude épistémique** | « Probabilités MC, entropie prédictive, entropie attendue et information mutuelle. » | `src/uncertainty/mc_dropout.py`, `src/uncertainty/decomposition.py`, `tests/test_uncertainty.py`, probabilités `[T,N,K]` exportées | ✅ fait |
| **#16 — [P1] Visualisation des prédictions incertaines** | « Exemples corrects/erronés, incertitudes et distributions. » | `scripts/visualiser_incertitude.py`, `results/figures/exemples_incertitude.png`, `results/figures/histogramme_mi.png`, rapport JSON avec indices et valeurs | ✅ fait |
| **#17 — [P2] Implémentation et comparaison des Deep Ensembles** | « Optionnel après validation MC Dropout ; seeds indépendantes et coût. » | 3 checkpoints (seeds 42/43/44, splits identiques), `scripts/run_ensembles.py`, `results/deep_ensembles_fashion-mnist.json` (comparaison à 30 passes) | ✅ fait |

### Grille de validation des tickets

La grille est identique sur les 6 tickets. État actuel (travail local, **aucun push ni PR effectué**) :

| Critère du ticket | État | Commentaire |
|---|---|---|
| Travail documenté et vérifié | ✅ | Documentation dans `docs/`, 11 tests unitaires verts, `compileall` OK. |
| Résultats ou preuves joints au ticket | ✅ dans le dépôt | JSON, figures et scripts dans `results/` et `scripts/`. À joindre aux issues lors de la PR vers `dev`. |
| Revue par un autre membre si changement de code | ⏳ | À faire : relecteur désigné par le contrat technique pour Elias = **Aminata**. |
| Intégration via PR regroupée vers `dev` si applicable | ⏳ | À faire après accord : PR regroupée vers `dev`, puis vers `main` après revue et tests. |

Les messages de commit référencent les numéros de tickets (#12 … #17) comme demandé dans le README.

## 2. Exigences obligatoires du PDF (sujet 5)

| Exigence du PDF | Couverture par cette branche | Responsable |
|---|---|---|
| Formulation claire du problème et justification du jeu de données | `docs/formulation-probabiliste.md`, `docs/compte-rendu-branche.md` (choix de Fashion-MNIST justifié), `results/data_manifest.json` | Elias (choix local) + Aminata (#5/#6) |
| Modélisation probabiliste explicite (vraisemblance, a priori) | `docs/formulation-probabiliste.md` : `Cat(softmax(f_w(x)))`, a priori gaussien, famille variationnelle Bernoulli, ELBO | Elias (#13) ✅ |
| Entraînement et description des choix d'implémentation | `scripts/train_baseline.py`, configs JSON par seed, 3 époques, Adam, dropout `p=0,25`, seeds fixées | Elias (#14) ✅ |
| Comparaison d'au moins deux approches d'incertitude | CNN classique vs MC Dropout vs Deep Ensembles, protocole et splits identiques, budgets de passes égaux | Elias (#14/#17) ✅ |
| Évaluation performances + qualité de l'incertitude + coût | Accuracy, NLL, entropie prédictive, entropie attendue, information mutuelle, temps train/inférence dans `results/*.json` | Elias (#15/#17) + Omar (#19/#23) |
| Calibration (diagramme de fiabilité, ECE, temperature scaling) | Interface prête : probabilités `[T,N,K]` et `[M,T,N,K]` exportées ; **non réimplémenté ici** | Omar (#20) ⏳ |
| Distribution shift / OOD | Non traité dans cette branche ; Failure cases à forte MI identifiés dans `results/visualisation_incertitude_fashion-mnist.json` comme point d'entrée | Omar (#21) ⏳ |
| Ensembles conformes et couverture empirique | Non traité dans cette branche | Omar (#22) ⏳ |
| Interprétation critique (hypothèses, limites, validité) | `docs/etat-art-bayesian-cnn.md` (limites de l'approximation), `docs/compte-rendu-branche.md` (limites expérimentales, non-comparabilité des anciens estimateurs corrigée par budget égal) | Elias ✅ + synthèse collective (#28) ⏳ |
| Reproductibilité du code | `requirements-probabilistic-models.txt`, `scripts/` complets, seeds, manifeste, tests, commandes dans le README | Elias ✅ + Gémima (#29) |

## 3. Ce qui reste hors de cette branche

- Le checkpoint baseline officiel d'Aminata n'est pas encore intégré ; le nôtre est un baseline local équivalent au contrat technique (logits `[N,C]`, dropout à l'entraînement). `scripts/run_mc_dropout.py --checkpoint <autre.pt>` accepte un checkpoint externe.
- Les métriques officielles de calibration et d'OOD (Omar) consomment directement les fichiers `results/probs_mc_fashion-mnist_test.npz` et `results/probs_ensemble_fashion-mnist_test.npz` (régénérables, ignorés par Git).
- Aucune expérience GPU ni CIFAR-10 n'a été réalisée : tous les chiffres sont CPU / Fashion-MNIST et journalisés avec leur device dans les JSON.
