# Intégration, reproductibilité et livraison

## Commandes disponibles maintenant (#24, #25, #29)
Après installation du package depuis la racine du dépôt :
```bash
python -m experiment.cli validate
python -m experiment.cli plan
```
`validate` accepte le dataset non décidé et vérifie les seeds, proportions, passes et paramètres. `plan` décrit six comparaisons (trois seeds × deux méthodes) et indique explicitement `blocked_dataset`. Ces commandes n'entraînent aucun modèle.

Après décision du ticket #5, renseigner `dataset` avec `CIFAR-10` ou `Fashion-MNIST`, mettre à jour le statut descriptif, puis committer la configuration. Depuis un checkout propre :
```bash
python -m experiment.cli record --output results/run-001/manifest.json
```
Le manifeste contient configuration, empreinte SHA-256, commit, versions installées et plan. Il refuse dataset=null, dépôt modifié, fichiers non suivis et écrasement d'un manifeste existant. Son statut est `prepared_not_executed` et ses métriques sont nulles : il ne prouve pas une exécution. Les artefacts dans results/ restent ignorés par Git. Vérifier manuellement que les dépendances sont installées (`pip check` et smoke test).

## Raccordement restant (#24, #25)
| Étape | Dépendance | Validation attendue |
|---|---|---|
| Préparation | #5–7, Aminata | source/licence, indices disjoints, mêmes class_names |
| Entraînement | #9–10, Aminata | checkpoint et normalisation traçables, aucune donnée test utilisée |
| MC Dropout | #14–15, Elias | dropout actif, batchnorm figée, [T,N,K] et moyenne corrects |
| Évaluation | #19–21, Omar | cas numériques de référence, calibration réservée, coût mesuré |
| Orchestration | #25, Gémima | enchaîner les implémentations réelles et arrêter sur erreur |

La CLI ne propose pas de commandes d'entraînement fictives. Ajouter les commandes prepare/train/evaluate quand leurs fonctions existent, puis un test d'intégration sur données synthétiques et une vérification sur dataset retenu. Les tests actuels couvrent contrats et préparation, pas les métriques ou modèles non implémentés.

## Préparation de livraison (#30–31)
1. Après intégration/revue, conserver les six branches et identifier le commit final sur main.
2. Exécuter installation propre Python 3.11, pip check, pytest, smoke test et pipeline réel pour les seeds prévues. Conserver les sorties et l'environnement résolu.
3. Vérifier rapport PDF de 8–12 pages hors annexes : problème, méthode, protocole, résultats réels, incertitude/calibration, robustesse, coûts, limites, bibliographie.
4. Constituer un ZIP depuis une liste explicite : code, configs validées, dépendances, README, rapport PDF, résultats/figures nécessaires et manifestes. Exclure .git, .venv, secrets, caches et téléchargements inutiles. Indiquer source/licence et méthode de récupération des données.
5. Ouvrir le ZIP et reproduire les commandes depuis un nouveau dossier. Vérifier absence de fichiers manquants et cohérence des chiffres avec le rapport.
6. Déposer officiellement le 5 décembre 2026 selon le canal du cours, conserver preuve/heure du dépôt. Soutenance le 7 décembre. Canal et heure limite exacte à confirmer dans les consignes du cours.

Aucune archive finale ni dépôt officiel n'est réalisé par cette préparation. #24, #25, #29, #30 et #31 restent ouverts jusqu'à leur validation complète.
