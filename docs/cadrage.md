# Cadrage scientifique — tickets #2 à #4

## Question et objectifs
À architecture et checkpoint identiques, MC Dropout fournit-il une incertitude plus utile qu'une prédiction déterministe pour identifier les erreurs et les entrées décalées, avec un coût d'inférence acceptable ? Étudier séparément performance de classification, calibration et détection d'incertitude.

Hypothèses à tester, sans présumer les résultats : H1, l'incertitude est plus élevée sur les erreurs et les données perturbées ; H2, le temperature scaling réduit la NLL sur données ID ; H3, augmenter le nombre de passes MC améliore la stabilité mais augmente le coût. Une hypothèse non confirmée reste un résultat scientifique valide.

## Périmètre et acceptation
Classification d'images supervisée, CNN entraîné avec dropout, comparaison déterministe/MC sur le même checkpoint. Deep Ensembles et prédiction conforme optionnels après le socle. Aucun déploiement métier requis. Dataset CIFAR-10 ou Fashion-MNIST : décision encore attendue d'Aminata au ticket #5 ; ne pas lancer une comparaison définitive avant cette décision.

Le travail sera accepté si les splits sont disjoints et traçables, les expériences partagent checkpoint et données, Accuracy/NLL/Brier/ECE et coûts sont exportés pour chaque seed, les limites sont discutées et une commande documentée reproduit les résultats. Aucun seuil d'amélioration arbitraire n'est imposé.

## Livraison
5 décembre 2026 : ZIP de code reproductible et rapport PDF de 8–12 pages hors annexes. 7 décembre : soutenance. Revue du cadrage, des interfaces et du protocole par l'équipe avant entraînements finaux. Le présent document constitue une proposition concrète à relire, pas une approbation collective.
