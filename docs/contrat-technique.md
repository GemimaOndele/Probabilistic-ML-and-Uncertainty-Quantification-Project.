# Contrats d'interface v1 — proposition à relire

## Données et modèle
`build_loaders(config) -> dict[str, DataLoader]` expose train, validation, calibration, test ; batch = `(images, labels, sample_ids)`. Images float32 [B,C,H,W], labels int64 [B] dans [0,K), identifiants stables et uniques par split. Les loaders d'évaluation ne mélangent pas l'ordre. Les dimensions C,H,W dépendent du dataset, encore non validé.

`build_model(config) -> torch.nn.Module` ; `forward(images) -> logits` float [B,K], sans softmax. Le CNN contient le dropout dès l'entraînement. Sauvegarder state_dict et métadonnées, sans objet Python arbitraire sérialisé.

## Prédictions et évaluation
`predict(model, loader, device) -> PredictionBatch` ; `predict_mc(model, loader, device, passes) -> PredictionBatch`. Champs : sample_ids, labels, probabilities [N,K], mc_probabilities optionnel [T,N,K]. La moyenne MC doit égaler probabilities. Pas de NaN/inf, valeurs dans [0,1], somme des lignes égale à 1 à 1e-6 près. Ordre des classes stable. `evaluation.contracts.validate_predictions` vérifie ces conditions sur des séquences Python ; les producteurs de tenseurs peuvent convertir via cpu().tolist().

`fit_temperature(logits, labels) -> float` utilise calibration uniquement ; pour MC conserver [T,N,K] et l'objectif défini au protocole. `compute_metrics(probabilities, labels, n_bins=15) -> dict[str,float]`. `compute_uncertainty(mc_probabilities) -> dict` fournit entropie prédictive, entropie attendue et information mutuelle par sample_id. Ces fonctions sont des signatures attendues, pas des implémentations déjà livrées.

Export JSON : schema_version=1, dataset_id, split, split_seed, training_seed, method, checkpoint_id, git_commit, config, environment, metrics, timing_seconds. Pour MC ajouter mc_passes. Les CSV de prédictions contiennent sample_id, target et p_0…p_K-1 ; passes MC dans un artefact séparé. Une ligne de synthèse ne doit pas mélanger seeds ou splits.

## Coordination
Aminata : données/CNN ; Elias : MC Dropout ; Omar : calibration/évaluation ; Gémima : intégration. Relecteurs convenus : Omar pour Gémima, Elias pour Aminata, Aminata pour Elias, Gémima pour Omar. Toute modification incompatible du contrat fait l'objet d'une discussion avant intégration. Protocole commun : protocole-experimental.md.
