import sys
import pandas as pd

sys.path.append("Fonctions")
from fonctions_Alfred import statistiques

# 1. Lire les données et les ranger dans l'ordre chronologique.
donnees = pd.read_csv("data/data_Alfred.csv")
donnees["date"] = pd.to_datetime(donnees["date"], utc=True)
donnees = donnees.set_index("date").sort_index()

# 2. Garder uniquement les 12 000 valeurs d'apprentissage.
apprentissage = donnees["prix"].iloc[:12000]

# 3. Calculer les variations à l'intérieur de l'apprentissage.
variations = apprentissage.diff().dropna()

# 4. Calculer les quatre caractéristiques sur les deux séries.
stats_brutes = statistiques(apprentissage)
stats_diff = statistiques(variations)

# 5. Construire un tableau facile à lire dans le terminal.
tableau = pd.DataFrame(
    [stats_brutes, stats_diff],
    index=["Serie brute", "Serie differenciee"]
)

print("Nombre de prix :", len(apprentissage))
print("Nombre de variations :", len(variations))
print()
print(tableau.round(4).to_string())

# 6. Enregistrer une ligne avec huit caractéristiques, comme demandé.
ligne = {}

for nom, valeur in stats_brutes.items():
    ligne[nom + "_brute"] = valeur

for nom, valeur in stats_diff.items():
    ligne[nom + "_diff"] = valeur

resultat = pd.DataFrame([ligne], index=["BTCUSDT_Alfred"])

resultat.to_csv(
    "Resultats/Resultats_Alfred/statistiques_Alfred.csv",
    sep=";",
    index_label="serie"
)