# Contrat technique proposé

Python/PyTorch, seed fixe, split train/validation/calibration/test disjoint, identique pour tous les modèles. Aminata fournit les chargeurs de données et un CNN retournant des logits [N,C]. Elias réutilise le checkpoint baseline avec dropout actif uniquement lors des passes MC, batchnorm en eval ; produit probabilités [T,N,C]. Omar calcule moyenne, entropie prédictive, information mutuelle, accuracy, NLL, Brier, ECE et coût ; temperature scaling ajusté exclusivement sur calibration. Gémima intègre les modules et scripts reproductibles.

Dossiers : src/data, src/models, src/uncertainty, src/evaluation, tests, configs, results, docs. Prévoir exports JSON/CSV avec seed, split, méthode, métriques, temps. Aucun résultat inventé. Choix du dataset et signatures détaillées à valider en équipe.

Relecteurs : Omar pour Gémima, Elias pour Aminata, Aminata pour Elias, Gémima pour Omar. PR vers dev par jalon, merge commits, pas de suppression de branches. Validation collective avant main.
