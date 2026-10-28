"code_Bahr_Chahine.py" contient le fruit du travail sur la prévision frugale de la consommation d'électricité en MW en France.
Pouvoir prédire une telle série permettrait de pouvoir alimenter correctement en électricité les Français via une source d'énergie typiquement démunie de stockage, notamment l'éolien.

Le code est réalisé sur python à l'aide des librairies numpy, pandas, matplotlib, sklearn, sys.
Ce code utilise principalement les fonctions pythons - ce qui le rend très compact - qu'il suffit d'appeler successivement selon le besoin dans le terminal via la commande :
python -c "import code_Bahr_Chahine as m; m.{nom_de_fonction}(m.{Série_d'entrée},{Autres arguments selon le besoin})"
Chacune des fonctions dispose de deux paramètres optionnels "graph" et "stats" initialisés à True, elles permettent d'afficher la courbe temporelle de la prédiction et de la réalité, ou bien de déterminer par une méthode graphique les hyperparamètres des modèles.

Il sera bon de noté qu'il s'agit là de mon premier projet de la sorte et que mes compétences en code sont restreintes (c'est bien pour cela que j'ai choisi ce mini projet), dès lors, certaines parties du code peuvent ne pas etre optimisées et ne tenir qu'a un fil (il m'a été difficile de modifier mes premières versions lorsque j'apprenais une nouvelle chose)