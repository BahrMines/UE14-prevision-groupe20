## Analyse des résultats – Série Vélib Denfert-Rochereau

Cette série porte sur le nombre de passages horaires de vélos au niveau de l'avenue Denfert-Rochereau, à Paris, sur la période 2020-2023 (17 544 observations).
Elle présente une moyenne de **37,05 passages par heure** et un écart type de **44,29**, soit un écart type relatif de près de 120 %.
Cette forte dispersion s'explique par une alternance entre des périodes de faible circulation, notamment la nuit, et des pics de fréquentation pouvant atteindre 326 passages horaires.

La kurtosis élevée (7,74) et l'asymétrie positive (1,98) témoignent également de la présence de pics importants et relativement rares.
Après différenciation, la kurtosis augmente à 11,42, ce qui traduit une distribution des variations encore plus concentrée autour de zéro, mais présentant quelques variations extrêmes.

Concernant les prévisions, le modèle de persistance obtient un nRMSE d'environ **45 %**, ce qui constitue une performance relativement limitée.
Cela peut s'expliquer par les variations horaires importantes du trafic cycliste, notamment aux heures de pointe, que la simple reprise de la dernière observation ne permet pas toujours d'anticiper.

L'AR(p) améliore sensiblement ces résultats, avec un nRMSE d'environ **30 %**. On observe sur la validation une diminution importante de l'erreur lorsque le nombre de retards augmente jusqu'à environ 25, puis une stabilisation progressive des performances.
Les observations supplémentaires apportent donc un gain plus limité au-delà de cette valeur.

L'ELM non différencié présente en revanche des résultats particulièrement décevants, avec un nRMSE supérieur à **80 %**, nettement moins bon que la persistance.
Malgré l'augmentation du nombre de neurones, les performances restent limitées. Cette difficulté peut notamment provenir de l'absence de normalisation des entrées : les valeurs élevées de la série peuvent entraîner une saturation de la fonction d'activation tanh, limitant ainsi la capacité d'apprentissage du réseau.

La différenciation améliore considérablement les résultats de l'ELM, dont le nRMSE descend à environ **35,5 %**.
L'AR différencié bénéficie également de cette transformation, avec un nRMSE d'environ **28,7 %**, constituant ainsi le meilleur résultat parmi les cinq modèles étudiés.
Cette amélioration suggère que les variations successives de la série sont plus faciles à modéliser que les valeurs brutes, notamment en raison des fluctuations importantes du trafic au cours de la journée.

Finalement, **l'AR différencié est le modèle le plus performant**, avec une erreur normalisée inférieure de près de 16 points à celle de la persistance.
L'ELM différencié parvient également à améliorer ses performances, mais reste moins précis que les modèles autorégressifs.
Ces résultats mettent en évidence l'intérêt de la différenciation pour cette série temporelle, ainsi que la pertinence d'une approche linéaire pour modéliser les variations du trafic cycliste.
