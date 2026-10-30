Analyse des résultats


Série de Malo : 
Cette série porte sur la pollution de l'air, et présente une kurtosis élevée(8.82) qui augmente très fortement avec la différenciation (77.1).
Cela témoigne de valeurs très élevées fréquentes, expliquant ainsi l'écart type élevé de 11.86 pour une grandeur positive de moyenne 21.78.
Concernant les résultats de prédiction, on obtient globalement des prédictions très performantes sûrement dûes à la forte inertie de la 
pollution de l'air. Le modèle ELM est moins performant que la persistance sur toutes les métriques, malgré une normalisation des entrées et 
une absence de surapprentissage: l'inertie de la pollution semble plus bénéficier à la persistance que la non linéarité de l'ELM. D'ailleurs, 
on peut observer sur la validation que augmenter le nombre de neurones L diminuait la performance, un modèle plus simple semble meilleur.
C'est également le cas pour le modèle ARp qui se dégrade avec p qui augmente, les retards additionnels semblent pertuber la prédiction. Enfin,
la différenciation augmente drastiquement l'efficacité pour l'ELM (44 points de nRMSE) et l'ARp (3.3 points de nRMSE demieux). On peut 
imaginer que la série était sujette à de fortes tendances (hebdomadaires avec les trajets maison-travail ou surtout saisonnier avec les 
pics de pollution estivaux). C'est finalement le modèle AR différencié qui est le plus performant sur toutes les métriques, avec 3.28% 
de nRMSE contre 25.99% pour la persistance, le gain de performance de prédiction est donc substantiel.



Série de Bahr :
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

Série de Tom :
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
