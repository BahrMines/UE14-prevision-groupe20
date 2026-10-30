## Présentation

Ce programme a pour objectif de comparer plusieurs méthodes de prévision à une heure d'échéance appliquées à une série temporelle de comptages horaires sur une piste cyclable (station Denfert-Rochereau).

Il implémente cinq modèles :
- **Persistance** : la dernière valeur observée est utilisée comme prévision.
- **AR(p)** : modèle autorégressif linéaire utilisant les \(p\) dernières observations.
- **ELM** : réseau de neurones à couche cachée unique (*Extreme Learning Machine*), dont les poids d'entrée sont initialisés aléatoirement et dont les poids de sortie sont estimés par régression ridge.
- **AR différencié** et **ELM différencié** : mêmes modèles appliqués aux variations successives de la série afin de ne prendre en compte que ses évolutions.

Les données sont réparties chronologiquement en trois ensembles : **apprentissage (70 %), validation (15 %) et test (15 %)**.
Les hyperparamètres \(p\) (nombre de retards) et \(L\) (nombre de neurones cachés) sont sélectionnés sur l'ensemble de validation.

Enfin, les performances des modèles sont comparées sur le jeu de test à l'aide de plusieurs métriques (nMBE, nMAE, nRMSE, \(R^2\) et NICE\(^\Sigma\)), ainsi que de graphiques confrontant les prévisions aux valeurs réellement observées.

## Organisation du code

Le code est organisé en plusieurs blocs de fonctions :
- **Métriques** : fonctions indépendantes permettant de calculer les indicateurs statistiques et les erreurs de prévision.
- **Construction des données** : création des matrices \(X\) et des vecteurs \(y\) à partir de la série temporelle et du nombre de retards \(p\).
- **Modélisation** : fonctions dédiées à l'apprentissage et à la prévision avec les modèles AR et ELM.
- **Optimisation des hyperparamètres** : recherche des meilleurs paramètres sur le jeu de validation.
- **Évaluation et visualisation** : comparaison des performances sur le jeu de test et tracé des prévisions.

Les fonctions sont conçues de manière modulaire : les étapes d'apprentissage, de prévision et d'évaluation sont séparées afin de faciliter les modifications et la réutilisation du code. 
Les calculs numériques s'appuient principalement sur NumPy, la manipulation des séries temporelles sur Pandas et leur représentation graphique sur Matplotlib.
