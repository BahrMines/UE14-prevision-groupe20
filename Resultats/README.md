### Analyse des résultats


**Série de Malo :** 
Cette série porte sur la pollution de l'air, et présente une kurtosis élevée(8.82) qui augmente très fortement avec la différenciation (77.1).
Cela témoigne de valeurs très élevées fréquentes, expliquant ainsi l'écart type élevé de 11.86 pour une grandeur positive de moyenne 21.78.
Concernant les résultats de prédiction, on obtient globalement des prédictions peu performantes sûrement dues à la grande kurtosis de la 
série. Le modèle ELM est moins performant que la persistance sur toutes les métriques, malgré une normalisation des entrées et 
une absence de surapprentissage: la non linéarité de l'ELM ne semble pas ici être un avantage. D'ailleurs, 
on peut observer sur la validation que augmenter le nombre de neurones L diminuait la performance, un modèle plus simple semble meilleur.
C'est également le cas pour le modèle ARp qui se dégrade avec p qui augmente, les retards additionnels semblent pertuber la prédiction, mais cela est presque sûrement une erreur de code ça ce phénomène est observé même sur les données d'apprentissage. Enfin,
la différenciation augmente drastiquement l'efficacité pour l'ELM (11 points de nRMSE de mieux) mais diminue l'efficacité de l'ARp (0.6 points de nRMSE de moins). On peut 
imaginer que la série était sujette à de fortes tendances (hebdomadaires avec les trajets maison-travail ou surtout saisonnier avec les 
pics de pollution estivaux). C'est finalement le modèle AR non différencié qui est le plus performant sur toutes les métriques, avec 25.19% 
de nRMSE contre 25.99% pour la persistance, le gain de performance de prédiction n'est donc pas substantiel, mais il existe.



**Série de Bahr :**
Nous nous interessons ici à la consommation (en MW) d'électricité en moyenne d'un foyer Français.
La série tirée de l'ODRE éCO2mix dont l'écart type relatif est de 14%, traduisant une consommation souvent élevée
à hauteur d'en moyenne 41497 MW par jour, dont les variations plutot faibles s'expliquent par le besoin énergétique variable
(plus fort à 19h en rentrant du travail qu'a 15h au bureau)
Il est à noter que les métriques sur mes séries me semblent aberrantes (ne convenant absolument pas aux graphes)
j'essaierai donc d'interpreter au mieux tout en me basant sur les representations temporelles.
La persistence présente une nRMSE de 20% ce qui est deja assez faible et peut s'expliquer par les lentes variations de la consommation,
avec une pente d'au plus 300MW/h. Concernant l'autoregression, le nRMSE décroit avec p qui augmente
par parcimonie on décide de prendre p=3 (cf graph). Lorsque l'on différencie, l'AR présente les meilleurs performances parmis les autre
modes de prévision avec une courbe se confondant pratiquement avec la variation réelle. L'ELM agit de manière similaire à l'AR, 
cependant le nRMSE suit une augmentation logarithmique avec p qui est optimal pour 2, pour une durée de prévision sur 2 jours, 20 couches
de neurones suffisent pour éviter le surapprentissage. A nouveau, la différenciation augmente largement les performances tout en restant 
moins bon que l'AR différencié

**Série de Tom :**
Cette série porte sur le nombre de passages horaires de vélos au niveau de l'avenue Denfert-Rochereau, à Paris, sur la période 2020-2023 (17 544 observations).
Elle présente une moyenne de 37,05 passages par heure et un écart type de 44,29, soit un écart type relatif de près de 120 %.
Cette forte dispersion s'explique par une alternance entre des périodes de faible circulation, notamment la nuit, et des pics de fréquentation pouvant atteindre 326 passages horaires.
La kurtosis élevée (7,74) et l'asymétrie positive (1,98) témoignent également de la présence de pics importants et relativement rares.
Après différenciation, la kurtosis augmente à 11,42, ce qui traduit une distribution des variations encore plus concentrée autour de zéro, mais présentant quelques variations extrêmes.
Concernant les prévisions, le modèle de persistance obtient un nRMSE d'environ 45 %, ce qui constitue une performance relativement limitée.
Cela peut s'expliquer par les variations horaires importantes du trafic cycliste, notamment aux heures de pointe, que la simple reprise de la dernière observation ne permet pas toujours d'anticiper.
L'AR(p) améliore sensiblement ces résultats, avec un nRMSE d'environ 30 %. On observe sur la validation une diminution importante de l'erreur lorsque le nombre de retards augmente jusqu'à environ 25, puis une stabilisation progressive des performances.
Les observations supplémentaires apportent donc un gain plus limité au-delà de cette valeur.
L'ELM non différencié présente en revanche des résultats particulièrement décevants, avec un nRMSE supérieur à 80 %, nettement moins bon que la persistance.
Malgré l'augmentation du nombre de neurones, les performances restent limitées. Cette difficulté peut notamment provenir de l'absence de normalisation des entrées : les valeurs élevées de la série peuvent entraîner une saturation de la fonction d'activation tanh, limitant ainsi la capacité d'apprentissage du réseau.
La différenciation améliore considérablement les résultats de l'ELM, dont le nRMSE descend à environ 35,5 %.
L'AR différencié bénéficie également de cette transformation, avec un nRMSE d'environ 28,7 %, constituant ainsi le meilleur résultat parmi les cinq modèles étudiés.
Cette amélioration suggère que les variations successives de la série sont plus faciles à modéliser que les valeurs brutes, notamment en raison des fluctuations importantes du trafic au cours de la journée.
Finalement, l'AR différencié est le modèle le plus performant, avec une erreur normalisée inférieure de près de 16 points à celle de la persistance.
L'ELM différencié parvient également à améliorer ses performances, mais reste moins précis que les modèles autorégressifs.
Ces résultats mettent en évidence l'intérêt de la différenciation pour cette série temporelle, ainsi que la pertinence d'une approche linéaire pour modéliser les variations du trafic cycliste.

**Série d’Alfred :**

Cette série correspond au prix de clôture horaire du Bitcoin en USDT sur Binance, de janvier 2022 à décembre 2024. Les 26 304 observations sont réparties chronologiquement entre apprentissage (12 000), validation (4 000) et test (10 304). Les modèles prévoient le prix une heure à l’avance.

Sur le test, la persistance, l’AR, l’AR différencié et l’ELM différencié donnent des nRMSE très proches de 0,574 %. Par conséquen, le dernier prix connu constitue donc déjà une bonne approximation du suivant, et les modèles apportent peu d’amélioration. C'est cela qui permet d'expliquer la quasi-superposition de leurs courbes. Cette faible erreur, normalisée par le prix moyen du Bitcoin, ne signifie pas pour autant que les modèles anticipent correctement les hausses et les baisses.

Cependant, l’ELM sans différenciation est nettement moins performant : en effet,sa nRMSE atteint 13,04 %, avec une nMBE de −8,04 %, qui nous indique une sous-estimation moyenne. Cela peut s’expliquer par les prix du test, souvent supérieurs à ceux rencontrés pendant l’apprentissage, et que ce modèle utilisant tanh extrapole difficilement. La différenciation permet de réduire cette difficulté en lui faisant prévoir les variations horaires plutôt que directement le niveau du prix.

L’ELM différencié obtient la meilleure nRMSE et l’AR différencié la meilleure nMAE, mais leurs gains restent infimes par rapport à la persistance. Finalement, celle-ci demeure donc une référence simple et efficace pour cette série.

**Tarik:**
Cette série porte sur la concentration horaire de NO₂ à la station Ajaccio Canetto, sur trois ans (de 2023 à 2026). Elle a une moyenne de 12,51 µg/m³ et un écart-type de 10,14, soit un écart-type relatif de 81 %, avec une asymétrie positive (1,62) et une kurtosis de 5,93 : les pics de pollution sont rares mais élevés. La différenciation garde 73 % de l’écart-type (7,40) et fait monter la kurtosis à 8,80, ce qui indique des sauts brusques d’une heure à l’autre, difficiles à anticiper à partir du passé de la série. La persistance obtient déjà un nRMSE de 47,2 % (R² de 0,44). Sur la validation, l’erreur de l’AR diminue quand p augmente jusqu’à 23, valeur choisie, proche du bord de la grille testée (1 à 24), et celle de l’ELM se stabilise autour de L = 100. Le meilleur modèle est l’ELM non différencié (nRMSE 40,4 %, R² de 0,59, NICEΣ de 0,868), soit 6,8 points de moins que la persistance, mais son avance sur l’AR (40,8 %) est faible. Pour l’AR, la différenciation améliore peu le nRMSE (40,6 %) mais supprime le biais (nMBE de −2,38 % à 0,08 %). Pour l’ELM, elle dégrade le résultat (41,4 %), et c’est le moins bon des quatre modèles appris. Le gain sur la persistance reste donc modeste (NICEΣ entre 0,87 et 0,89) : les pics, sous-estimés par tous les modèles, dépendent probablement de facteurs extérieurs comme le trafic ou le vent, absents de la série.

**Dylan**
Projet de Prévision : Séries Temporelles de la Température à Paris
Objectif : modéliser et  prévoir l'évolution de la température horaire de l'air à Paris. Les données météorologiques exploitées sont issues de l'API Open-Meteo Historical Weather (coordonnées : Latitude 48.8534, Longitude 2.3488) et couvrent la période allant de début 2022 à fin 2024. Ces données sont libres d'utilisation et redistribuées sous la licence Creative Commons Attribution 4.0 (CC BY 4.0).

On découpe notre série en 3 parties d'environ un an chacun : 8 760 heures pour la phase d'apprentissage (ajustement des poids des algorithmes), 8 760 heures pour la validation (optimisation des hyperparamètres de complexité et de mémoire) et 8 760 heures pour le jeu de test final servant à l'évaluation en conditions réelles.

Une analyse physique et statistique préalable met en évidence la forte inertie thermique de la série brute, caractérisée par un écart-type de 7.44°C. Afin d'isoler la dynamique à court terme de la température, une différenciation temporelle a été appliquée. Cette opération mathématique rend le signal stationnaire : la moyenne s'effondre autour de 0°C et l'écart-type chute drastiquement à 1.01°C, prouvant que les variations thermiques d'une heure à l'autre sont minimes et structurellement plus stables pour un algorithme d'apprentissage.

Pour évaluer l'efficacité des prévisions, plusieurs modèles ont été implémentés et comparés à un modèle naïf de persistance, lequel établit le score de base avec un nRMSE de 7.22%. Sur les données brutes, un modèle autorégressif linéaire AR(4) abaisse cette erreur à 5.82%. L'introduction de non-linéarité via un réseau de neurones ELM classique (200 neurones) permet de descendre à 5.66%.

Toutefois, les meilleures performances sont obtenues en aidant les modèles avec des caractéristiques physiques ou mathématiques. D'une part, l'ELM enrichi par un encodage circulaire du calendrier (fonctions sinus et cosinus pour l'heure et le jour) permet au réseau de capter l'impact direct du cycle solaire, atteignant le meilleur score absolu avec un nRMSE de 5.07%. D'autre part, l'application d'un modèle AR(24) sur la série différenciée, suivie d'une étape de recoloration finale, offre une performance redoutable (5.12%), démontrant qu'il est souvent plus efficace de modéliser les micro-variations plutôt que la tendance globale.

L'exécution du projet nécessite un environnement Python standard équipé des bibliothèques numpy, pandas et matplotlib. Le lancement de l'étude complète s'effectue en exécutant le script principal depuis la racine du projet.

voir code/Dylan/readme pour plus de detail et pour les figures.