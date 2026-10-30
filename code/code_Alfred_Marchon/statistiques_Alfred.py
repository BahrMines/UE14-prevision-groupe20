import sys
import pandas as pd

sys.path.append("Fonctions")
from fonctions_Alfred import statistiques

# On lit les données et on les range dans l'ordre chronologique.
donnees=pd.read_csv("data/data_Alfred.csv")
donnees["date"]=pd.to_datetime(donnees["date"], utc=True)
donnees=donnees.set_index("date").sort_index()

# On garde uniquement les 12 000 valeurs d'apprentissage.
apprentissage=donnees["prix"].iloc[:12000]

# On calcule ensuite les variations à l'intérieur de l'apprentissage.
variations=apprentissage.diff().dropna()

# On calcule les quatre caractéristiques sur les deux séries.
stats_brutes=statistiques(apprentissage)
stats_diff=statistiques(variations)

# On construit un tableau 
tableau = pd.DataFrame(
    [stats_brutes, stats_diff],
    index=["Serie brute", "Serie differenciee"])

print("Nombre de prix :",len(apprentissage))
print("Nombre de variations :",len(variations))
print()
print(tableau.round(4).to_string())

#On enregistre une ligne avec huit caractéristiques
ligne={}

for nom,valeur in stats_brutes.items():
    ligne[nom+"_brute"]=valeur
for nom, valeur in stats_diff.items():
    ligne[nom+"_diff"]=valeur

resultat=pd.DataFrame([ligne],index=["BTCUSDT_Alfred"])

resultat.to_csv(
    "Resultats/Resultats_Alfred/statistiques_Alfred.csv",
    index_label="serie")