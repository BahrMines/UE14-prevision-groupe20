import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# On indique où se trouve notre fichier de fonctions.
sys.path.append("Fonctions")
from fonctions_Alfred import persistance, bilan_erreurs

# On lit les données.
donnees=pd.read_csv("data/data_Alfred.csv")
donnees["date"]=pd.to_datetime(donnees["date"], utc=True)
donnees=donnees.set_index("date").sort_index()

x = donnees["prix"]

# On découpe la série dans l'ordre chronologique.
apprentissage=x.iloc[:12000]
validation=x.iloc[12000:16000]
test=x.iloc[16000:]

# L’apprentissage servira à estimer les coefficients des futurs modèles AR et ELM.
# La validation va servir à choisir leurs réglages et à examiner leur fonctionnement
# Et le test servira à mesurer les performances finales après ces choix

print("Apprentissage :",len(apprentissage))
print("Validation :",len(validation))
print("Test réservé :",len(test))

# Les valeurs évaluées doivent être de vraies observations.
if donnees["valeur_completee"].iloc[12000:].any():
    raise ValueError("Une valeur à évaluer a été complétée : à vérifier.")
# On calculer la persistance sur l'apprentissage et la validation.
previsions=persistance(x.iloc[:16000])

# On garde les prévisions correspondant aux 4000 observations de validation.
prevision_validation=previsions.iloc[12000:16000]
# On créée un tableau du type date | mesure | prevision
resultats = pd.DataFrame({
    "mesure":validation,
    "prevision":prevision_validation})
# Puis on calcule les erreurs sur la validation et on rajoute la colonne erreur 
resultats["erreur"]=resultats["prevision"]-resultats["mesure"]

# On calcule les cinq indicateurs avec les fonctions communes
scores = bilan_erreurs(
    resultats["mesure"],
    resultats["prevision"],
    resultats["prevision"])

tableau_scores=pd.DataFrame(
    [scores],index=["Persistance"])

print(resultats.head())
print("\nScores sur la validation :")
print(tableau_scores.round(4).to_string())

tableau_scores.to_csv(
    "Resultats/Resultats_Alfred/scores_persistance_validation.csv",index_label="modele")

# On enregistrer les résultats.
resultats.to_csv("Resultats/Resultats_Alfred/persistance_Alfred.csv")

# et finalement on trace les 72 premières heures de validation soit 3 jours 
resultats[["mesure", "prevision"]].iloc[:72].plot(figsize=(10, 4))

plt.title("Persistance : 72 premières heures de validation")
plt.xlabel("Date (UTC)")
plt.ylabel("Prix du Bitcoin (USDT)")
plt.tight_layout()
plt.savefig("Resultats/Resultats_Alfred/persistance_Alfred.png")
plt.show()