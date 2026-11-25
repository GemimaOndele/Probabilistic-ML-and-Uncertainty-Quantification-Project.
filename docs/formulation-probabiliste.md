# Formulation mathématique et probabiliste

Mission GitHub : **#13 — [P0] Formulation mathématique et probabiliste** (branche `feature/probabilistic-models`).
Vraisemblance catégorielle, distribution prédictive et approximation variationnelle pour le sujet 5 (classification d'images avec incertitude épistémique).

## 1. Données et vraisemblance

Jeu de données supervisé `D = {(x_i, y_i)}_{i=1..N}` avec `x_i ∈ R^{C×H×W}` (Fashion-MNIST : `1×28×28`, CIFAR-10 : `3×32×32`) et `y_i ∈ {1,...,K}`, `K = 10`.

Le réseau `f_w` produit des logits `z_i = f_w(x_i) ∈ R^K`. La vraisemblance est catégorielle :

```
p(y_i = k | x_i, w) = softmax(z_i)_k = exp(z_{i,k}) / Σ_j exp(z_{i,j})
log p(D | w) = Σ_i log softmax(f_w(x_i))_{y_i}
```

L'entraînement classique applique une régularisation L2 (weight decay) et minimise la `-log p(D|w)` ; il fournit un unique estimateur `ŵ` (MAP approximatif).

## 2. Approche bayésienne et approximation variationnelle

On place une a priori sur les poids `p(w)` (gaussienne nulle de variance `τ^{-1}` dans la formulation de Gal & Ghahramani). L'inférence bayésienne exige le postérieur :

```
p(w | D) = p(D | w) p(w) / p(D)
```

intraitable pour un CNN. On introduit une famille variationnelle paramétrée par les taux de dropout `q(w) = ∏_ij q(w_ij)`. Pour un poids multiplicatif par un masque de Bernoulli `γ_ij ~ Bern(1-p)` :

```
w_ij = ŵ_ij · γ_ij
q(γ_ij = 1) = 1 - p
```

L'ELBO (evidence lower bound) à maximiser est :

```
L(q) = Σ_i E_{q(w)}[ log p(y_i | f_w(x_i)) ] - KL(q(w) || p(w))
```

Gal & Ghahramani montrent que l'objectif d'un réseau entraîné avec dropout et régularisation L2 est un estimateur stochastique de cette ELBO (avec un choix de prior adapté). Chaque minibatch correspond à un tirage des masques : l'entraînement optimise donc `q`, et non un point unique.

**Conséquence concrète** : pour que l'inférence soit cohérente avec l'entraînement, le dropout doit rester actif pendant les passes MC, tandis que BatchNorm utilise ses statistiques courantes (`model.eval()` puis réactivation des `Dropout`). Un checkpoint entraîné *sans* dropout ne permet pas de MC Dropout valide : les masques n'ayant pas été appris, l'incertitude estimée serait artefactuelle.

## 3. Distribution prédictive

Pour une nouvelle image `x*`, la prédictive bayésienne marginalise sur les poids :

```
p(y* | x*, D) = ∫ p(y* | x*, w) p(w | D) dw
```

MC Dropout approxime cette intégrale par moyenne de Monte Carlo sur `T` masques `γ_t ~ q` (poids enregistrés `ŵ` fixes) :

```
p̂(y* = k | x*, D) = (1/T) Σ_{t=1..T} softmax(f_{ŵ ⊙ γ_t}(x*))_k
```

On utilise la moyenne des **probabilités** (pas des logits ni des labels). `T` est une constante de la configuration, journalisée avec les résultats. Le prédicteur MAP correspondant est `argmax_k p̂_k`.

## 4. Décomposition de l'incertitude

Soit `p_t = softmax(f_{ŵ ⊙ γ_t}(x*))`. On définit :

```
H_préd = - Σ_k p̂_k log p̂_k                         entropie prédictive (incertitude totale)
H_moy  = (1/T) Σ_t [ - Σ_k p_{t,k} log p_{t,k} ]    entropie attendue (incertitude aléatoire)
MI     = H_préd - H_moy                             information mutuelle (incertitude épistémique)
```

Identité (Kendall & Gal) :

```
I[y* ; w | x*, D] = H[y* | x*, D] - E_{q(w)}[ H[y* | x*, w] ]
```

Interprétation sur le sujet 5 :

- `H_préd` grande + `MI` faible → l'image est ambiguë de façon intrinsèque (classes visuellement proches, bruit de label) : incertitude aléatoire.
- `MI` grande → les masques de dropout désaccordent : le modèle manque de connaissance pour cet exemple (cas typique d'un exemple atypique ou OOD) : incertitude épistémique.
- `MI ≈ 0` pour `T = 1` : l'information mutuelle ne peut pas être estimée avec une seule passe ; il faut `T > 1`.

Les formules sont implémentées dans `src/uncertainty/mc_dropout.py` et testées dans `tests/test_uncertainty.py`.

## 5. Calibration (interface avec la mission d'Omar)

Le softmax moyen `p̂` n'est pas garanti calibré. Le temperature scaling applique `softmax(z / τ)` avec `τ > 0` appris sur le **split de calibration uniquement**, puis évalué sur le split de test. Le même τ est appliqué aux `T` passes avant moyennage. Les métriques officielles (Accuracy, NLL, Brier, ECE) et les diagrammes de fiabilité sont calculés par Omar dans `src/evaluation` : notre branche produit les tableaux de probabilités `[T, N, K]` et les indicateurs d'incertitude nécessaires, sans dupliquer ces métriques.

## 6. Protocole et reproductibilité

- Seed globale fixée (Python, NumPy, PyTorch), seeds indépendantes pour les membres du Deep Ensembles (mission #17).
- Splits train / validation / calibration / test disjoints et identiques pour toutes les méthodes, conformément au contrat technique.
- Journaux d'exécution au format JSON/CSV avec `seed`, `split`, `méthode`, `dataset`, `T`, métriques et temps (train + inference).
- Aucun résultat inventé : les chiffres publiés dans `results/` proviennent des scripts exécutés dans ce dépôt.
