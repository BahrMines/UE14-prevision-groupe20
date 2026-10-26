import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys 

# Tambouille d'IA pour importer mes fonctions...
racine_projet = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. On l'ajoute au chemin de recherche de Python
if racine_projet not in sys.path:
    sys.path.insert(0, racine_projet)
from Fonctions.Fonctions_statistiques import nMAE,nRMSE,nMBE,RSquared,NICE,afficher_statistiques

# Importation des données en sautant les 3 premières lignes, on renomme les colonnes et on convertit
df = pd.read_csv("data/temperature_Paris.csv", sep=",", skiprows=3)
df.columns = ["date_heure", "temperature"]
df["date_heure"] = pd.to_datetime(df["date_heure"])
df["temperature"] = df["temperature"].interpolate(method="linear")
x = df["temperature"].values


#On sépare apprentissage, valdiation ,test,on a 27000 valeurs donc on separe en 26 280 valeurs
n_train, n_val = 8760, 8760

x_train = x[:n_train]#Valeurs d'apprentissage
x_val = x[n_train : n_train + n_val]#Valeurs de valdiation pour p de Ar(p) puis ELM
x_test = x[n_train + n_val :]

#On mesure quelques stats sur nos valeurs d'entrainements,bruts puis différenciée 
print("=== ÉTAPE 1 : STATISTIQUES (Apprentissage) ===")
y_train_diff = np.diff(x_train)  # Série différenciée y_t = x_t - x_{t-1}

afficher_statistiques(x_train, "Série Brute (x)")
afficher_statistiques(y_train_diff, "Série Différenciée (y)")

# ==============================================================================
# 1-MODÈLE DE PERSISTANCE ET ÉVALUATION SUR LE TEST
# ==============================================================================
print("=== ÉTAPE 2 : MODÈLE DE PERSISTANCE ===")

# Prévision à horizon 1 : la prédiction pour t+1 est la valeur en t
y_true = x_test[1:]
y_pred_pers = x_test[:-1]

# Évaluation avec les métriques du groupe
print("--- Métriques d'erreur sur le jeu de Test ---")
print(f"nMBE  : {nMBE(y_true, y_pred_pers)*100:.5f} %")
print(f"nMAE  : {nMAE(y_true, y_pred_pers)*100:.5f} %")
print(f"nRMSE : {nRMSE(y_true, y_pred_pers)*100:.5f} %")
print(f"R²    : {RSquared(y_true, y_pred_pers):.5f}")

# Le NICE prend en paramètre (Vraies_valeurs, Pred_Modele, Pred_Persistance)
# Ici, comme on évalue la persistance elle-même son score vaudra 1 par déf.
nice_score = NICE(y_true, y_pred_pers, y_pred_pers)
print(f"NICE  : {nice_score:.5f}")

# ==============================================================================
# 2. MODÈLE AR(p) - CHOIX DE P PAR APPROCHE ENVELOPPE (VALIDATION)
# ==============================================================================
print("=== ÉTAPE 3 : SÉLECTION DE L'ORDRE p POUR LE MODÈLE AR(p) ===")

# 1. Centrage des données avec la moyenne de l'apprentissage UNIQUEMENT
mu = np.mean(x_train)
xc_train = x_train - mu
xc_val = x_val - mu

# Fonction pour créer les matrices X (retards) et Y (cible)
def creer_matrices_AR(serie_centree, p):
    N = len(serie_centree)
    X = np.zeros((N - p, p))
    Y = serie_centree[p:]
    for i in range(p):
        X[:, i] = serie_centree[p - 1 - i : N - 1 - i]
    return X, Y

# 2. Test des p de 1 à 24 sur la validation
erreurs_validation = []
p_max_test = 24

for p in range(1, p_max_test + 1):
    # Entraînement sur le Train (Moindres carrés)
    X_train, Y_train = creer_matrices_AR(xc_train, p)
    # np.linalg.lstsq résout l'équation : Y = X * w
    w, _, _, _ = np.linalg.lstsq(X_train, Y_train, rcond=None)
    
    # Prévision sur la Validation
    # On a besoin des p dernières valeurs du train pour prédire les premières de la validation
    xc_concat = np.concatenate((xc_train[-p:], xc_val))
    X_val, Y_val = creer_matrices_AR(xc_concat, p)
    
    Y_pred_val = X_val @ w
    
    # Calcul du RMSE sur la validation
    rmse_val = np.sqrt(np.mean((Y_val - Y_pred_val)**2))
    erreurs_validation.append(rmse_val)

# 3. Application de la règle de parsimonie (seuil de 0,5% du meilleur)
meilleur_rmse = min(erreurs_validation)
seuil_parsimonie = meilleur_rmse * 1.005

# On cherche le premier p (le plus petit) qui passe sous le seuil
p_optimal = 1
for i, rmse in enumerate(erreurs_validation):
    if rmse <= seuil_parsimonie:
        p_optimal = i + 1
        break

print(f"Meilleur RMSE absolu sur validation : {meilleur_rmse:.4f} °C")
print(f"Seuil de tolérance (parsimonie +0.5%) : {seuil_parsimonie:.4f} °C")
print(f"--> Ordre AR optimal retenu : p = {p_optimal}")
# ==============================================================================
# 5. ÉVALUATION DU MODÈLE AR(24) SUR LE JEU DE TEST
# ==============================================================================
print("\n=== ÉTAPE 4 : ÉVALUATION FINALE DU MODÈLE AR(24) ===")

p_final = 24

# 1. Entraînement définitif de l'AR(24) sur tout l'apprentissage centré
X_train_final, Y_train_final = creer_matrices_AR(xc_train, p_final)
w_final, _, _, _ = np.linalg.lstsq(X_train_final, Y_train_final, rcond=None)

# 2. Préparation des données de Test
# Pour prédire les premières valeurs du Test, on a besoin des 24 dernières de la Validation
xc_test_concat = np.concatenate((xc_val[-p_final:], x_test - mu))
X_test, Y_test_c = creer_matrices_AR(xc_test_concat, p_final)

# 3. Prédiction et dé-centrage (+ mu)
Y_pred_c = X_test @ w_final
y_pred_AR = Y_pred_c + mu
y_true_test = Y_test_c + mu  # Correspond exactement à x_test

# 4. Génération de la Persistance pour la fonction NICE
# La persistance pour x_test[0] est x_val[-1], puis x_test[0] pour x_test[1], etc.
pers_test = np.concatenate((x_val[-1:], x_test[:-1]))

# 5. Calcul des métriques avec vos fonctions
print("--- Métriques d'erreur de l'AR(24) sur le Test ---")
print(f"nMBE  : {nMBE(y_true_test, y_pred_AR):.5f}")
print(f"nMAE  : {nMAE(y_true_test, y_pred_AR):.5f}")
print(f"nRMSE : {nRMSE(y_true_test, y_pred_AR):.5f}")
print(f"R²    : {RSquared(y_true_test, y_pred_AR):.5f}")

nice_score_AR = NICE(y_true_test, y_pred_AR, pers_test)
print(f"NICE  : {nice_score_AR:.5f}")

# ==============================================================================
# 6. VISUALISATION GRAPHIQUE (1ère Semaine de Test)
# ==============================================================================
jours_a_afficher = 7
heures = jours_a_afficher * 24

plt.figure(figsize=(12, 6))
plt.plot(y_true_test[:heures], label="Vraies températures (x)", color="black", linewidth=2)
plt.plot(y_pred_AR[:heures], label="Prédictions AR(24)", color="red", linestyle="--", linewidth=1.5)
plt.plot(pers_test[:heures], label="Persistance", color="blue", linestyle=":", linewidth=1, alpha=0.6)

plt.title(f"Prévision de température à Paris - 1ère semaine de Test (2024)")
plt.xlabel("Heures")
plt.ylabel("Température (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()