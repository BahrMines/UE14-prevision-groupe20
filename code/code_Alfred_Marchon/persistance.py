import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# On indique où se trouve notre fichier de fonctions.
sys.path.append("Fonctions")
from fonctions_Alfred import persistance

# 1. Lire les données.
donnees = pd.read_csv("data/data_Alfred.csv")
donnees["date"] = pd.to_datetime(donnees["date"], utc=True)
donnees = donnees.set_index("date").sort_index()

x = donnees["prix"]

# 2. Découper la série dans l'ordre chronologique.
apprentissage = x.iloc[:12000]
validation = x.iloc[12000:16000]
test = x.iloc[16000:]

# L’apprentissage servira à estimer les coefficients des futurs modèles AR et ELM.
# La validation servira à choisir leurs réglages et à examiner leur fonctionnement.
# Le test servira à mesurer les performances finales après ces choix.

print("Apprentissage :", len(apprentissage))
print("Validation :", len(validation))
print("Test réservé :", len(test))

# Les valeurs évaluées doivent être de vraies observations.
if donnees["valeur_completee"].iloc[12000:].any():
    raise ValueError("Une valeur à évaluer a été complétée : à vérifier.")

# 3. Calculer la persistance sur l'apprentissage et la validation.
previsions = persistance(x.iloc[:16000])

# 4. Garder les prévisions correspondant aux 4000 observations de validation.
prevision_validation = previsions.iloc[12000:16000]

# On créée un tableau du type date | mesure | prevision
resultats = pd.DataFrame({
    "mesure": validation,
    "prevision": prevision_validation
})

# 5. Calculer les erreurs sur la validation et on rajoute la colonne erreur 
resultats["erreur"] = resultats["prevision"] - resultats["mesure"]

# On calcule la MAE: l'erreur absolue moyenne 
# Dans notre série, la MAE s’exprime en USDT
mae = np.mean(np.abs(resultats["erreur"]))

# On calcule la RMSE, la racine de la moyenne des erreurs au carré.
rmse = np.sqrt(np.mean(resultats["erreur"] ** 2))

# la division permet d’exprimer l’erreur relativement au niveau moyen
nrmse = 100 * rmse / validation.mean()

print(resultats.head())

# On prend deux chiffres après la virgule 
print("MAE de validation :", round(mae, 2), "USDT")
print("RMSE de validation :", round(rmse, 2), "USDT")

# On prend quatre chiffres après la virgule parce que l'erreur relative est plus petite.
print("nRMSE de validation :", round(nrmse, 4), "%")

# 6. Enregistrer les résultats.
resultats.to_csv("Resultats/Resultats_Alfred/persistance_Alfred.csv", sep=";")

# 7. Tracer les 72 premières heures de validation soit 3 jours 
resultats[["mesure", "prevision"]].iloc[:72].plot(figsize=(10, 4))

plt.title("Persistance : 72 premières heures de validation")
plt.xlabel("Date (UTC)")
plt.ylabel("Prix du Bitcoin (USDT)")
plt.tight_layout()
plt.savefig("Resultats/Resultats_Alfred/persistance_Alfred.png")
plt.show()