# Veille scientifique — Bayesian CNN et MC Dropout

Références, hypothèses et limites de l'approximation bayésienne retenue pour le sujet 5.
Mission GitHub : **#12 — [P0] Veille scientifique : Bayesian CNN et MC Dropout** (branche `feature/probabilistic-models`).

## Pourquoi une approche bayésienne ici

Un CNN classique produit un softmax `softmax(f_w(x))` pour un unique jeu de poids `w`. Ce softmax est souvent interprété à tort comme une probabilité de confiance : il ne distingue pas l'incertitude **aléatoire** (bruit intrinsèque des données, ex. deux images visuellement proches mais de classes différentes) de l'incertitude **épistémique** (méconnaissance du modèle, ex. image hors distribution). Le sujet 5 demande explicitement de quantifier l'incertitude épistémique et de comparer au moins deux approches.

## Références principales

| Référence | Apport pour ce projet |
|---|---|
| Gal & Ghahramani, *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*, ICML 2016 ([arXiv:1506.02142](https://arxiv.org/abs/1506.02142)) | Montre que l'entraînement avec dropout est une approximation variationnelle d'un posterior bayésien sur les poids. L'inférence MC Dropout consiste à garder le dropout actif à l'inférence et moyenner `T` passes. |
| Gal, *Uncertainty in Deep Learning*, thèse de doctorat, Cambridge 2016 ([PDF](https://www.cs.cam.ac.uk/~yg245/pubs/thesis.pdf)) | Détails de l'ELBO, de la famille variationnelle de Bernoulli et de la décomposition entropie / information mutuelle. |
| Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*, NeurIPS 2017 ([arXiv:1703.04977](https://arxiv.org/abs/1703.04977)) | Distingue incertitude aléatoire et épistémique en vision par ordinateur ; l'information mutuelle est proposée comme proxy de l'incertitude épistémique. |
| Blundell et al., *Weight Uncertainty in Neural Networks*, ICML 2015 ([arXiv:1505.05424](https://arxiv.org/abs/1505.05424)) | Bayes by Backprop : alternative explicite (postérieur gaussien) à l'approximation par dropout, plus coûteuse. |
| Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017 ([arXiv:1612.01474](https://arxiv.org/abs/1612.01474)) | Deep Ensembles : plusieurs réseaux indépendants (seeds différentes), référence robuste mais coût d'entraînement multiplié. |
| Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599)) | Température ajustée sur un split de calibration séparé ; ECE et diagrammes de fiabilité (mission d'Omar). |
| Ovadia et al., *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*, NeurIPS 2019 ([arXiv:1906.02530](https://arxiv.org/abs/1906.02530)) | Les méthodes d'incertitude se dégradent sous distribution shift ; souligne l'importance du protocole OOD (mission d'Omar). |
| Xiao, Rasul & Vollgraf, *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, 2017 ([arXiv:1708.07747](https://arxiv.org/abs/1708.07747)) | Jeu de données principal candidat : 70 000 images 28×28, 10 classes, split officiel 60k/10k. |
| Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 ([rapport](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)) | CIFAR-10 : 60 000 images 32×32 couleur, 10 classes ; candidat alternatif / OOD. |

## Hypothèses retenues

1. **Vraisemblance catégorielle** : `y | x, w ~ Cat(softmax(f_w(x)))` (mission #13).
2. **Famille variationnelle** : masques de dropout Bernoulli sur les poids, taux `p` fixé ; les autres poids sont déterministes.
3. **BatchNorm déterministe à l'inférence** : `model.eval()` puis réactivation manuelle des seules couches `Dropout` pendant les `T` passes, conformément au contrat technique. Les statistiques courantes de BatchNorm sont utilisées ; on n'échantillonne pas des statistiques de batch pendant MC.
4. **Séparation des splits** : calibration utilisée uniquement par Omar pour le temperature scaling ; les métriques de notre branche sont calculées sur le split de test.
5. **Réutilisation du checkpoint baseline** : le modèle est entraîné avec dropout actif (sinon les masques sont sans effet à l'inférence) ; le checkpoint contient architecture + poids + seed + configuration.

## Limites et précautions

- MC Dropout n'échantillonne **pas** le vrai posterior : la famille variationnelle est restrictive (Bernoulli indépendant, taux constant) et l'ELBO est optimisée de façon approximative (SGD, minibatchs). Les incertitudes peuvent être sous-estimées.
- L'information mutuelle estimée sur `T` passes finies a une erreur Monte Carlo ; on fixe `T` dans la config et on journalise sa valeur (avec `T=1` l'incertitude épistémique estimée est nulle par construction).
- Le softmax moyen et la moyenne des logits ne sont pas équivalents ; nous utilisons la **moyenne des probabilités** après softmax, comme dans l'équation de predictive distribution.
- Le dropout appliqué avant BatchNorm ou avec des statistiques de batch stochastiques peut injecter du bruit parasite ; d'où le choix `model.eval()` + dropout forcé.
- Deep Ensembles (mission #17) améliore généralement la couverture mais multiplie le coût par le nombre de membres ; à comparer à coût égal, pas seulement à précision égale.
- Sur Fashion-MNIST / CIFAR-10 en distribution, l'incertitude épistémique est souvent faible ; l'intérêt principal apparaît sur OOD ou sur exemples ambigus (mission #16).

## Ce que ce projet produit

- `docs/formulation-probabiliste.md` : formulation mathématique (mission #13).
- `src/uncertainty/mc_dropout.py` : probabilités MC + entropies + information mutuelle (missions #14/#15).
- `scripts/train_baseline.py`, `scripts/run_mc_dropout.py` : pipeline reproductible.
- `scripts/visualiser_incertitude.py` : exemples corrects/erronés et distributions (mission #16).
- `scripts/run_ensembles.py` : seeds indépendantes et coût (mission #17).
