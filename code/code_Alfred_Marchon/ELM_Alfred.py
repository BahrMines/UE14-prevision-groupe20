import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("Fonctions")
from fonctions_Alfred import prevision_elm, persistance, nrmse, bilan_erreurs

# Toujours la même chose 
donnees =pd.read_csv("data/data_Alfred.csv")
donnees["date"] =pd.to_datetime(donnees["date"], utc=True)
x= donnees.set_index("date").sort_index()["prix"]
apprentissage =x.iloc[:12000]
validation =x.iloc[12000:16000]
historique =x.iloc[:16000]

# On met le même nombre de retards que pour l'AR brut retenu.
p= 5
scores= []
meilleurs=[]
previsions_par_L= {}

# Cinq tailles de réseau et vingt tirages pour chaque taille.
for L in [5, 20, 50, 100, 500]:
    meilleur_score =float("inf")
    for graine in range(20):
        previsions= prevision_elm(apprentissage, historique, p, L, graine)
        score = nrmse(validation, previsions)
        scores.append({"L": L, "graine": graine, "nRMSE_validation (%)": score})

        if score < meilleur_score:
            meilleur_score =score
            meilleure_graine =graine
            previsions_par_L[L]= previsions

    meilleurs.append({
        "L": L,
        "graine": meilleure_graine,
        "nRMSE_validation (%)": meilleur_score})
    print("L =", L, "- meilleure nRMSE :", round(meilleur_score, 6), "%", flush=True)

# Le meilleur tirage de chaque taille est conservé.
# Puis on choisit la plus petite taille à moins de 0,01 % du meilleur score.
meilleurs = pd.DataFrame(meilleurs)
seuil = 1.0001 * meilleurs["nRMSE_validation (%)"].min()
choix = meilleurs[meilleurs["nRMSE_validation (%)"] <= seuil].iloc[0]
L_choisi =int(choix["L"])
graine_choisie = int(choix["graine"])
print("\nELM retenu : p =", p, ", L =", L_choisi,", graine =",graine_choisie)

# On réutilise les fonctions d'évaluation communes.
resultats = pd.DataFrame(index=validation.index)
resultats["mesure"] = validation
resultats["ELM"] =previsions_par_L[L_choisi]
resultats["persistance"] =persistance(historique).loc[validation.index]
resultats["erreur_ELM"] =resultats["ELM"]-resultats["mesure"]

bilans = {}
for modele in ["ELM", "persistance"]:
    bilans[modele] =bilan_erreurs(resultats["mesure"], resultats[modele], resultats["persistance"])
tableau_scores = pd.DataFrame(bilans).T
print("\nScores sur la validation :")
print(tableau_scores.round(6).to_string())

# On conserve tous les essais et les paramètres retenus pour la suite.
dossier = "Resultats/Resultats_Alfred/"
pd.DataFrame(scores).to_csv(dossier + "essais_ELM.csv",index=False)
pd.DataFrame([{
    "p": p, "L": L_choisi, "graine": graine_choisie,
    "alpha": 0.001, "tolerance_relative": 0.0001}]).to_csv(dossier +"parametres_ELM.csv",index=False)
resultats.to_csv(dossier+ "ELM_validation.csv")
tableau_scores.to_csv(dossier +"scores_ELM_validation.csv",index_label="modele")

resultats[["mesure","persistance", "ELM"]].iloc[:72].plot(
    figsize=(10, 4), style=["b-", "g-", "r--"])
plt.title(f"ELM : p={p},L={L_choisi}, graine={graine_choisie}")
plt.xlabel("Date (UTC)")
plt.ylabel("Prix du Bitcoin (USDT)")
plt.tight_layout()
plt.savefig(dossier +"ELM_validation.png")
plt.show()