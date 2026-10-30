import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("Fonctions")
from fonctions_Alfred import prevision_ar, persistance, nrmse, bilan_erreurs

# On lit les données dans l'ordre chronologique.
donnees =pd.read_csv("data/data_Alfred.csv")
donnees["date"]= pd.to_datetime(donnees["date"], utc=True)
donnees = donnees.set_index("date").sort_index()
x =donnees["prix"]

# On garde exactement le même découpage que pour la persistance.
apprentissage= x.iloc[:12000]
validation= x.iloc[12000:16000]
historique= x.iloc[:16000]

# On teste les 25 premières valeurs de p 
scores =[]
previsions_par_p ={}

for p in range(1, 25):
    previsions= prevision_ar(apprentissage, historique, p)
    score =nrmse(validation, previsions)
    scores.append({
        "p": p,
        "nRMSE_validation (%)": score})
    previsions_par_p[p]= previsions

scores= pd.DataFrame(scores)
print(scores.round(6).to_string(index=False))

# Tolérance relative de 0,01 % au-dessus du meilleur score.
seuil =1.0001 * scores["nRMSE_validation (%)"].min()
candidats =scores[scores["nRMSE_validation (%)"] <= seuil]
p_choisi =int(candidats.iloc[0]["p"])

print("\nOrdre retenu :",p_choisi)

# On peut comparer AR et persistance sur les mêmes heures de validation.
resultats = pd.DataFrame(index=validation.index)
resultats["mesure"]= validation
resultats["AR"] =previsions_par_p[p_choisi]
resultats["persistance"]=persistance(historique).loc[validation.index]
resultats["erreur_AR"] =resultats["AR"] -resultats["mesure"]

bilans ={}
for modele in ["AR", "persistance"]:
    bilans[modele] =bilan_erreurs(
        resultats["mesure"],
        resultats[modele],
        resultats["persistance"])

tableau_scores =pd.DataFrame(bilans).T

print("\nScores sur la validation :")
print(tableau_scores.round(4).to_string())

tableau_scores.to_csv("Resultats/Resultats_Alfred/scores_AR_validation.csv",index_label="modele")

# Finalement on enregistrer les résultats et le graphique
scores.to_csv("Resultats/Resultats_Alfred/choix_p_AR.csv",index=False)

resultats.to_csv(
    "Resultats/Resultats_Alfred/AR_validation.csv")
resultats[["mesure", "persistance", "AR"]].iloc[:72].plot(
    figsize=(10, 4), style=["b-", "g-", "r--"])

plt.title(f"AR({p_choisi}) et persistance : 72 heures de validation")
plt.xlabel("Date (UTC)")
plt.ylabel("Prix du Bitcoin (USDT)")
plt.tight_layout()
plt.savefig("Resultats/Resultats_Alfred/AR_validation.png")
plt.show()

ecart=resultats["AR"] -resultats["persistance"]

print("Écart maximal AR/persistance :",ecart.abs().max(), "USDT")