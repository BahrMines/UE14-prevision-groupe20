import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("Fonctions")
from fonctions_Alfred import persistance, prevision_ar, prevision_elm, bilan_erreurs

# On lit les données et on retrouve nos trois périodes.
donnees =pd.read_csv("data/data_Alfred.csv")
donnees["date"] =pd.to_datetime(donnees["date"], utc=True)
x = donnees.set_index("date").sort_index()["prix"]
apprentissage = x.iloc[:12000]
test =x.iloc[16000:]
variations= x.diff().dropna()
apprentissage_diff =variations.iloc[:11999]

# On fixe les réglages sur la validation,[4000:] retire ses 4 000 prévisions
resultats = pd.DataFrame({"mesure": test})
resultats["Persistance"] = persistance(x).loc[test.index]
resultats["AR"]=prevision_ar(apprentissage, x, p=5)[4000:]
resultats["ELM"] =prevision_elm(apprentissage, x, p=5, L=20, graine=14)[4000:]

# On prévoit les variations puis on les ajoute au dernier prix connu
dernier_prix=resultats["Persistance"].to_numpy()
variation_AR =prevision_ar(apprentissage_diff, variations, p=4, positif=False)[4000:]
variation_ELM = prevision_elm(
    apprentissage_diff, variations, p=4, L=20, graine=3, positif=False)[4000:]
resultats["AR diff"] =np.maximum(0, dernier_prix + variation_AR)
resultats["ELM diff"] =np.maximum(0, dernier_prix + variation_ELM)

# On compare enfin les cinq modèles avec nos fonctions communes.
modeles =["Persistance", "AR", "ELM", "AR diff", "ELM diff"]
bilans ={}
for modele in modeles:
    bilans[modele] =bilan_erreurs(test, resultats[modele], resultats["Persistance"])
scores= pd.DataFrame(bilans).T
print("Scores sur les", len(test),"heures de TEST :")
print(scores.round(6).to_string())

dossier ="Resultats/Resultats_Alfred/"
scores.to_csv(dossier +"comparaison_test.csv",index_label="modele")
resultats.to_csv(dossier+ "previsions_test.csv")

# On montre les prix et les écarts entre modèles sur les 24 premières heures pour
# pouvoir y voir quelque chose et on utilisde différents styles
extrait =resultats.iloc[:24]
ecarts= extrait[modeles].subtract(extrait["Persistance"], axis=0)
fig, axes= plt.subplots(2, 1, figsize=(12, 8), sharex=True)
extrait.plot(ax=axes[0],style=["k-o", "c-", "b--", "r-.", "g:", "m--"])
ecarts.plot(ax=axes[1], style=["c-", "b--", "r-.", "g:", "m--"], marker=".")
axes[0].set_title("24 premières heures du test : mesures et cinq modèles")
axes[0].set_ylabel("Prix (USDT pour 1 BTC)")
axes[1].set_title("Différences entre prévisions, pas erreurs par rapport aux mesures")
axes[1].set_ylabel("Prévision − persistance (USDT)")
axes[1].set_xlabel("Date (UTC)")
for ax in axes:
    ax.grid(alpha=0.3)
    ax.legend(ncol=3)
plt.tight_layout()
plt.savefig(dossier + "comparaison_test_24h.png", dpi=150)
plt.show()