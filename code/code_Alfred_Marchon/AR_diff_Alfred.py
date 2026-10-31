import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("Fonctions")
from fonctions_Alfred import prevision_ar, persistance, nrmse, bilan_erreurs

# On Lit les prix et on conserve le même découpage.
donnees =pd.read_csv("data/data_Alfred.csv")
donnees["date"]= pd.to_datetime(donnees["date"], utc=True)
x =donnees.set_index("date").sort_index()["prix"]

historique= x.iloc[:16000]
validation= x.iloc[12000:16000]

# Puis on construit les variations, sans inclure la phase test
#historique contient l’apprentissage et la validation, sans le test
#diff() calcule chaque prix moins le prix précédent
#dropna() enlève la première variation qui est impossible à calculer
variations =historique.diff().dropna()
apprentissage_diff =variations.iloc[:11999]

# Pour chaque date cible, le dernier prix connu est celui de l'heure précédente
dernier_prix =persistance(historique).loc[validation.index]

# On teste les ordres de 1 à 24, avec des erreurs calculées sur les prix
scores =[]
previsions_par_p ={}

for p in range(1, 25):
    variations_prevues = prevision_ar(
        apprentissage_diff, variations, p, positif=False)
    prix_prevus = np.maximum(0, dernier_prix.to_numpy() +variations_prevues)
    score = nrmse(validation, prix_prevus)

    scores.append({"p": p, "nRMSE_validation (%)": score})
    previsions_par_p[p]= prix_prevus

scores =pd.DataFrame(scores)
print(scores.round(6).to_string(index=False))

# On prend toujours la même tolérance relative de 0,01 % que pour l'AR 
seuil =1.0001 * scores["nRMSE_validation (%)"].min()
candidats= scores[scores["nRMSE_validation (%)"] <= seuil]
p_choisi= int(candidats.iloc[0]["p"])
print("\nOrdre retenu pour AR différencié :", p_choisi)

# On évalue les prix reconstruits avec les fonctions communes
resultats = pd.DataFrame(index=validation.index)
resultats["mesure"] =validation
resultats["AR_diff"]= previsions_par_p[p_choisi]
resultats["persistance"]= dernier_prix
resultats["erreur_AR_diff"]= resultats["AR_diff"]-resultats["mesure"]

bilans ={}
for modele in ["AR_diff", "persistance"]:
    bilans[modele] =bilan_erreurs(resultats["mesure"],resultats[modele], resultats["persistance"])

tableau_scores =pd.DataFrame(bilans).T
print("\nScores sur la validation :")
print(tableau_scores.round(6).to_string())

# On récupère les résultats
dossier= "Resultats/Resultats_Alfred/"
scores.to_csv(dossier +"choix_p_AR_diff.csv",index=False)
resultats.to_csv(dossier +"AR_diff_validation.csv")
tableau_scores.to_csv(dossier + "scores_AR_diff_validation.csv",index_label="modele")

resultats[["mesure", "persistance", "AR_diff"]].iloc[:72].plot(figsize=(10, 4), style=["b-", "g-", "r--"])
plt.title(f"AR différencié ({p_choisi}) et persistance")
plt.xlabel("Date (UTC)")
plt.ylabel("Prix du Bitcoin (USDT)")
plt.tight_layout()
plt.savefig(dossier + "AR_diff_validation.png")
plt.show()
