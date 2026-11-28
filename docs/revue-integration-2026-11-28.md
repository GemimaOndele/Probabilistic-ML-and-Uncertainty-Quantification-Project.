# Préparation de l'intégration — 28 novembre 2026

Révisions lues : PR #34 `9c08992`, PR #35 `379c7dd`. Les CI de ces révisions réussissent. Les résultats publiés par Elias sont des résultats déclarés dans sa branche, pas une reproduction indépendante dans ce contrôle. Les PR restent non fusionnées.

| Point observé | Conséquence | Action avant comparaison finale |
|---|---|---|
| Elias exporte `probs_mc`, Omar charge `mc_probs` | Évaluation directe impossible | Adaptateur `experiment.artifacts normalize` fourni dans #33 |
| Exports MC sans sample_ids | Impossible de vérifier l'ordre et le non-chevauchement | Exporter les identifiants réels, stables par dataset et split officiel ; ne pas inventer des IDs à partir de l'ordre des lignes |
| Scripts Elias importent `src.*`, package #33 installé comme modules de premier niveau | Deux conventions d'import à réunifier | Tester les scripts depuis la racine après intégration locale ; conserver les commandes existantes jusqu'à décision commune |
| requirements #35 non figé et concurrent au fichier #33 | Conflit de dépendances à résoudre | Conserver le socle figé et ajouter les dépendances réellement nécessaires, puis vérifier l'installation propre |
| Elias utilise Fashion-MNIST et 50000/5000/5000 ; #33 propose 80/10/10 et dataset=null | Protocoles différents ; #5 toujours ouvert | Aminata valide le dataset et les indices communs ; ne pas modifier rétroactivement les résultats existants |
| #35 ajuste température puis conforme sur la même calibration | La garantie conforme standard n'est pas justifiée pour ce réemploi | Réserver une calibration conforme indépendante, ou désactiver cette partie pour les résultats finaux en attendant correction |
| Les résultats des différentes seeds partagent des noms de sortie côté évaluation | Risque d'écrasement si même dossier | Utiliser un dossier distinct par dataset/seed/checkpoint/méthode |

## Commandes disponibles dans #33
```bash
python -m experiment.artifacts normalize results/legacy.npz results/canonical.npz
python -m experiment.artifacts check-pair results/calibration.npz results/test.npz
```
La normalisation accepte exactement une clé de prédictions parmi probs_mc/mc_probs/probs/logits, vérifie formes, probabilités, finitude et labels, puis conserve les valeurs. Elle ne génère ni labels ni identifiants et refuse d'écraser la cible.

Le contrôle de paire exige labels et sample_ids uniques et disjoints, avec même représentation et nombre de classes. Les identifiants doivent être globaux (par exemple dataset + split officiel + index d'origine). Des index locaux repartant de zéro par sous-split ne suffisent pas. L'ordre des classes, le checkpoint et l'origine des splits doivent encore être vérifiés via les manifestes. Ce contrôle de format ne prouve pas à lui seul l'absence de fuite dans l'entraînement.

## Ordre de suite proposé
1. Faire relire #33 par Omar et résoudre les conventions communes.
2. Valider avec Aminata #5–7 et le checkpoint partagé. Le travail de la branche data-baseline n'est pas encore publié au moment de ce contrôle.
3. Tester localement la combinaison #33/#34/#35, résoudre les conflits, vérifier les nouveaux tests puis obtenir les revues avant fusion dans dev.
4. Exporter calibration et test de chaque méthode, contrôler leurs identifiants et évaluer dans des dossiers distincts.
5. Consolider uniquement des résultats reproductibles dans le rapport et conserver les limites des expériences provisoires.

## Vérifications effectuées dans cette mise à jour
35 tests du socle d'intégration passent localement. Un essai séparé avec le véritable chargeur `load_predictions` de la révision #35 `379c7dd` reproduit le rejet de `probs_mc`, puis charge avec succès l'archive normalisée. Les probabilités et labels sont vérifiés inchangés avec NumPy. Cet essai utilise des données synthétiques et ne reproduit pas les expériences scientifiques.
