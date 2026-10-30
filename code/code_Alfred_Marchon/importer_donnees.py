import pandas as pd

# On récupère les prix horaires, mois par mois.
mois_telecharges=[]

for annee in [2022, 2023, 2024]:
    for mois in range(1,13):
        print("Téléchargement :",annee,mois)

        url = (
            "https://data.binance.vision/data/spot/monthly/klines/"
            f"BTCUSDT/1h/BTCUSDT-1h-{annee}-{mois:02d}.zip")
        # Colonne 0 : début de l'heure ; colonne 4 : prix de clôture.
        # On enlève les autres colonnes inutiles 
        tableau=pd.read_csv(
            url, compression="zip", header=None, usecols=[0, 4])
        # On renomme les deux colonnes 
        tableau.columns=["date", "prix"]
        mois_telecharges.append(tableau)

# On rassemble tous les mois dans un seul tableau.
# ignore index pour créer un nouvelle numérotation globale des lignes
donnees=pd.concat(mois_telecharges, ignore_index=True)

# On date donc le prix au moment où l’heure vient de se terminer.
donnees["date"] = (pd.to_datetime(donnees["date"], unit="ms", utc=True)
    + pd.Timedelta(hours=1))
donnees=donnees.set_index("date").sort_index()

# On rétablit les heures manquantes dans le calendrier.
# On demande un calendrier avec une ligne toutes les heure
donnees = donnees.asfreq("h")
donnees["valeur_completee"]=donnees["prix"].isna()

# Une heure isolée manquante est remplacée par le prix précédent.
donnees["prix"]=donnees["prix"].ffill(limit=1)
if donnees["prix"].isna().any():
    raise ValueError("Il reste des données manquantes : à vérifier.")

# On enregistre les données pour les prochains calculs.
donnees.to_csv("data/data_Alfred.csv")

print("Nombre de valeurs :",len(donnees))
print("Heures complétées :",donnees["valeur_completee"].sum())
print(donnees.head())