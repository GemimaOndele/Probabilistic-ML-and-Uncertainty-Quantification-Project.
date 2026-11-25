# Architecture fonctionnelle et technique

## Flux fonctionnel
Collecter et explorer les images → figer les splits → entraîner et sélectionner le CNN → produire logits déterministes et passes MC → calibrer → mesurer performance, incertitude et robustesse → consolider rapport et figures. Chaque étape conserve la provenance des artefacts et échoue explicitement si un contrat est invalide.

## Modules et responsabilités
| Dossier | Responsable | Entrée → sortie |
|---|---|---|
| src/data | Aminata | images + manifeste → batches et indices de splits |
| src/models | Aminata | tenseurs images → logits ; entraînement → checkpoint |
| src/uncertainty | Elias | checkpoint + batches → probabilités MC et incertitudes |
| src/evaluation | Omar | probabilités + labels → métriques, calibration, figures |
| configs, scripts, tests | Gémima | configuration → orchestration, contrôles et exports |
| results | équipe | artefacts locaux traçables, aucun résultat inventé |

Le package utilise les dossiers src/data, src/models, src/uncertainty et src/evaluation déjà convenus. Les signatures sont détaillées dans contrat-technique.md. Les implémentations scientifiques restent aux tickets spécialisés ; le socle expose uniquement la validation du contrat et la seed Python.

## Artefacts
Manifeste dataset : source/version/licence, class_names ordonnés, sample_id stable et empreinte des données. Splits : indices disjoints et seed. Checkpoint : state_dict, paramètres d'architecture, class_names, normalisation et seed. Prédictions : sample_id, labels et probabilités conservant exactement l'ordre du manifeste. Résultats : schéma JSON versionné, commit, configuration, versions, matériel, durées et métriques. Les fichiers volumineux restent hors Git ; publier un manifeste et un emplacement accessible pour la livraison.

Les six branches sont permanentes. Synchroniser depuis dev ; PR groupée par jalon vers dev, merge commit après revue, validation collective avant main. Ne pas supprimer ces branches.
