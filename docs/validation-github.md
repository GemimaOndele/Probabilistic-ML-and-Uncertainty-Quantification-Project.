# Vérification indépendante — 25 novembre 2026

- 32 issues #1 à #32 retrouvées via le connecteur GitHub.
- Six branches présentes : main, dev et les quatre feature/* décrites au README ; aucune branche supprimée.
- main et dev signalées protected=true par l'API de branches.
- Détail des protections : accès refusé HTTP 403 au connecteur ; exigences de revue et checks non certifiées par ce contrôle.
- Droits des collaborateurs et champs/vues/dates du Projects #7 : non revérifiés dans ce contrôle API. Le ticket #1 reste ouvert tant que ces éléments ne sont pas vérifiés.
- CI initiale présente ; cette PR ajoute installation du package, tests de contrats et smoke test PyTorch, à contrôler dans Actions après publication.

Cette vérification ne vaut pas validation définitive de toute la configuration.
