import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# 0. CONFIG ET IMPORTATION
# ==============================================================================
# L'IA a fait une tambouille pour importer les fonctions métriques 
racine_projet = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if racine_projet not in sys.path:
    sys.path.insert(0, racine_projet)

from Fonctions.Fonctions_statistiques import nMAE, nRMSE, nMBE, RSquared, NICE, afficher_statistiques

# On importe les données format csv et on renomme les colonnes
# skiprows=3 supprime l'en-tête inutile du fichier CSV qui crée des erreurs
df = pd.read_csv("data/temperature_Paris.csv", sep=",", skiprows=3)
df.columns = ["date_heure", "temperature"]

# conversion de la colonne temporelle en vrai format "Date" compréhensible par Python
df["date_heure"] = pd.to_datetime(df["date_heure"])

#Normalement il y a toute les valeurs mais je peux pas verifier 28k valeurs
#donc au cas ou on interpole linéairement les données manquantes
df["temperature"] = df["temperature"].interpolate(method="linear")

# Extraction des valeurs brutes dans un tableau numpy 1D (vecteur)
x = df["temperature"].values


#On découpe nos 28k données ( 3 ans de température)
n_train, n_val = 8760, 8760
x_train = x[:n_train]                         # Année 1 (Entraînement)
x_val = x[n_train : n_train + n_val]          # Année 2 (Choix des paramètres p et L)
x_test = x[n_train + n_val :]                 # Année 3 (Examen final)

# ==============================================================================
# 1. STATISTIQUES 
# ==============================================================================
print("=== ÉTAPE 1 : STATISTIQUES ===")
#np.diff calcule y_t = x_t - x_{t-1} sur tout le tableau
y_train_diff = np.diff(x_train)
#on le fait sur la série brute et différenciée et on en fera l'interprétation
afficher_statistiques(x_train, "Série Brute (x)")
afficher_statistiques(y_train_diff, "Série Différenciée (y)")

# ==============================================================================
# 2. MODÈLE DE PERSISTANCE
# ==============================================================================
print("\n=== ÉTAPE 2 : MODÈLE DE PERSISTANCE ===")
#on veut coder x̂t+1 = xt
# ICI pas d'apprentissage. On simule la prévision sur le jeu de test.
# x_test[:-1] prend tout le test sauf la dernière heure.
# x_val[-1:] prend la toute dernière heure de la validation (pour prédire la 1ère heure du test).
# np.concatenate les "colle" ensemble pour créer un tableau de prédictions décalé d'une heure.
pers_test = np.concatenate((x_val[-1:], x_test[:-1]))
y_true_test = x_test # La réalité à deviner

print("--- Métriques Persistance (Test) ---")
print(f"nMBE  : {nMBE(y_true_test, pers_test)*100:.5f} %")
print(f"nMAE  : {nMAE(y_true_test, pers_test)*100:.5f} %")
print(f"nRMSE : {nRMSE(y_true_test, pers_test)*100:.5f} %")
print(f"R²    : {RSquared(y_true_test, pers_test):.5f}")
print(f"NICE  : {NICE(y_true_test, pers_test, pers_test):.5f}")

# ==============================================================================
# 3. MODÈLE AR(p) : VALIDATION ET TEST (donnée centrée)
# ==============================================================================
print("\n=== ÉTAPE 3 : MODÈLE AR(p) ===")

# on centre : on retire la moyenne d'apprentissage pour que la série oscille autour de 0
#on fait comme le diapo en gros
mu = np.mean(x_train) 
xc_train = x_train - mu
xc_val = x_val - mu
xc_test = x_test - mu

def creer_matrices_AR(serie_centree, p):
    """
    Transforme un vecteur 1D en matrice X (les p retards) et vecteur Y (la cible).
    Si on a [a, b, c, d, e] et p=2 :
    X = [[a, b],    Y = [c,
         [b, c],         d,
         [c, d]]         e]
    Pour crée la matrice Y = X w  du diapo
    """
    N = len(serie_centree)
    X = np.zeros((N - p, p))
    Y = serie_centree[p:]
    for i in range(p):
        X[:, i] = serie_centree[p - 1 - i : N - 1 - i]
    return X, Y

erreurs_validation_AR = []
liste_p = list(range(1, 25))

# BOUCLE DE VALIDATION : On cherche le meilleur p
for p in liste_p:
    X_train_AR, Y_train_AR = creer_matrices_AR(xc_train, p)
    
    # résolution par Moindres Carrés Ordinaires, on utilise numpy (lstsq = Least Squares)
    # mathématiquement, calcule : w = (X^T X)^-1 X^T Y
    w, _, _, _ = np.linalg.lstsq(X_train_AR, Y_train_AR, rcond=None)
    
    #on reconstruit les matrices de validation (avec le bout du "train" pour faire la jointure)
    xc_concat_val = np.concatenate((xc_train[-p:], xc_val))
    X_val_AR, Y_val_AR = creer_matrices_AR(xc_concat_val, p)
    
    # ici on fait laprédiction : Y_pred = X * w (Multiplication matricielle donc @ en Python)
    Y_pred_val = X_val_AR @ w
    
    # on calcul manuel du RMSE sur la validation pour ce p
    erreurs_validation_AR.append(np.sqrt(np.mean((Y_val_AR - Y_pred_val)**2)))

# ENTRAÎNEMENT FINAL sur p=24
p_final = 24
X_train_final, Y_train_final = creer_matrices_AR(xc_train, p_final)
w_final, _, _, _ = np.linalg.lstsq(X_train_final, Y_train_final, rcond=None)

# PRÉVISION SUR LE TEST
xc_test_concat = np.concatenate((xc_val[-p_final:], xc_test))
X_test_AR, Y_test_AR_c = creer_matrices_AR(xc_test_concat, p_final)

# On prédit (X @ w) et on ajoute 'mu' pour "dé-centrer" et revenir aux vrais degrés Celsius
y_pred_AR = (X_test_AR @ w_final) + mu

print(f"--- Métriques AR({p_final}) (Test) ---")
print(f"nMBE  : {nMBE(y_true_test, y_pred_AR)*100:.5f} %")
print(f"nMAE  : {nMAE(y_true_test, y_pred_AR)*100:.5f} %")
print(f"nRMSE : {nRMSE(y_true_test, y_pred_AR)*100:.5f} %")
print(f"R²    : {RSquared(y_true_test, y_pred_AR):.5f}")
print(f"NICE  : {NICE(y_true_test, y_pred_AR, pers_test):.5f}")

# ==============================================================================
# 4. MODÈLE ELM : OPTIMISATION ET TEST (Normalisation mu + sigma)
# ==============================================================================
print("\n=== ÉTAPE 4 : MODÈLE ELM (Normalisé) ===")

#  x_norm = (x - moyenne) / ecart_type pour normaliser la matrice 
# et c'est indispensable pour éviter que la fonction tanh() des neurones ne sature à -1 ou +1
sigma = np.std(x_train)
xn_train = (x_train - mu) / sigma
xn_val = (x_val - mu) / sigma
xn_test = (x_test - mu) / sigma

alpha = 1e-3 # Paramètre de régularisation pour éviter que les poids explosent( cf diapo)
liste_L = [5, 10, 20, 50, 100, 200, 500]
nb_tirages = 20

X_train_ELM, Y_train_ELM = creer_matrices_AR(xn_train, p_final)
xn_val_concat_ELM = np.concatenate((xn_train[-p_final:], xn_val))
X_val_ELM, Y_val_ELM = creer_matrices_AR(xn_val_concat_ELM, p_final)

meilleur_rmse_global = float('inf')
rmse_par_L = []
meilleur_L, meilleur_W_in, meilleur_b, meilleur_beta = None, None, None, None

for L in liste_L:
    meilleur_rmse_pour_ce_L = float('inf')
    for _ in range(nb_tirages):
        
        # tirage aléatoire des poids d'entrée (W_in) et des biais (b) selon une loi normale
        W_in = np.random.normal(0, 1, (p_final, L))
        b = np.random.normal(0, 1, L)
        
        # couche cachée de l'apprentissage : H = tanh(X * W_in + b)
        H_train = np.tanh(X_train_ELM @ W_in + b)
        
        # résolution alpha : beta_hat = (H^T H + alpha * I)^-1 H^T Y comme le diapo
        # .T est la matrice transposée, np.eye(L) est la matrice identité (I)
        # np.linalg.solve(A, B) calcule A^-1 * B de manière plus stable numériquement
        I = np.eye(L)
        beta = np.linalg.solve(H_train.T @ H_train + alpha * I, H_train.T @ Y_train_ELM)
        
        # Évaluation sur la validation
        H_val = np.tanh(X_val_ELM @ W_in + b)
        
        # Y_pred = H * beta. On re-multiplie par sigma et ajoute mu pour repasser en °C réels
        Y_pred_val_celsius = (H_val @ beta) * sigma + mu
        Y_val_celsius = Y_val_ELM * sigma + mu
        rmse_val = np.sqrt(np.mean((Y_val_celsius - Y_pred_val_celsius)**2))
        
        if rmse_val < meilleur_rmse_pour_ce_L:
            meilleur_rmse_pour_ce_L = rmse_val
            W_in_opt, b_opt, beta_opt = W_in, b, beta
            
    rmse_par_L.append(meilleur_rmse_pour_ce_L)
    if meilleur_rmse_pour_ce_L < meilleur_rmse_global:
        meilleur_rmse_global = meilleur_rmse_pour_ce_L
        meilleur_L, meilleur_W_in, meilleur_b, meilleur_beta = L, W_in_opt, b_opt, beta_opt

print(f"--> ELM retenu : L = {meilleur_L} (RMSE Val = {meilleur_rmse_global:.4f} °C)")

# PRÉVISION SUR LE TEST (Avec les meilleurs W_in, b et beta)
xn_test_concat_ELM = np.concatenate((xn_val[-p_final:], xn_test))
X_test_ELM, _ = creer_matrices_AR(xn_test_concat_ELM, p_final)

H_test = np.tanh(X_test_ELM @ meilleur_W_in + meilleur_b)
# Dé-normalisation finale : (Prédiction * écart-type) + moyenne
y_pred_ELM = (H_test @ meilleur_beta) * sigma + mu

print(f"--- Métriques ELM (Test) ---")
print(f"nMBE  : {nMBE(y_true_test, y_pred_ELM)*100:.5f} %")
print(f"nMAE  : {nMAE(y_true_test, y_pred_ELM)*100:.5f} %")
print(f"nRMSE : {nRMSE(y_true_test, y_pred_ELM)*100:.5f} %")
print(f"R²    : {RSquared(y_true_test, y_pred_ELM):.5f}")
print(f"NICE  : {NICE(y_true_test, y_pred_ELM, pers_test):.5f}")

# ==============================================================================
# 5. MODÈLE ELM + SIN/COS (L'heure et le jour)
# ==============================================================================
print("\n=== ÉTAPE 5 : MODÈLE ELM + CALENDRIER (SIN/COS) ===")

# Extraction numérique des attributs de temps
heures = df["date_heure"].dt.hour.values
jours = df["date_heure"].dt.dayofyear.values

# Encodage trigonométrique (Circulaire) : permet à l'IA de comprendre que l'heure 23 est voisine de l'heure 0
sin_h, cos_h = np.sin(2 * np.pi * heures / 24), np.cos(2 * np.pi * heures / 24)
sin_j, cos_j = np.sin(2 * np.pi * jours / 365), np.cos(2 * np.pi * jours / 365)
# np.column_stack assemble ces 4 vecteurs 1D en une matrice à 4 colonnes
C = np.column_stack((sin_h, cos_h, sin_j, cos_j))

C_train, C_val, C_test = C[:n_train], C[n_train : n_train + n_val], C[n_train + n_val :]

def creer_matrices_ELM_sincos(serie_normalisee, matrice_C, p):
    """
    Fait la même chose que l'AR mais ajoute la matrice de calendrier 'C' 
    à la fin des p retards. La matrice X gagne 4 colonnes.
    """
    N = len(serie_normalisee)
    X_retards = np.zeros((N - p, p))
    for i in range(p):
        X_retards[:, i] = serie_normalisee[p - 1 - i : N - 1 - i]
    # np.hstack (Horizontal Stack) colle la matrice temporelle C à droite de la matrice X
    return np.hstack((X_retards, matrice_C[p:])), serie_normalisee[p:]

X_train_sc, Y_train_sc = creer_matrices_ELM_sincos(xn_train, C_train, p_final)
C_val_concat = np.concatenate((C_train[-p_final:], C_val))
X_val_sc, Y_val_sc = creer_matrices_ELM_sincos(xn_val_concat_ELM, C_val_concat, p_final)

p_total = p_final + 4 # On a désormais 28 entrées (24 retards + 4 variables temps)
meilleur_rmse_sc = float('inf')
L_sc, W_in_sc, b_sc, beta_sc = None, None, None, None

for L in liste_L:
    meilleur_rmse_L = float('inf')
    for _ in range(nb_tirages):
        W_in = np.random.normal(0, 1, (p_total, L))
        b = np.random.normal(0, 1, L)
        H_train = np.tanh(X_train_sc @ W_in + b)
        I = np.eye(L)
        beta = np.linalg.solve(H_train.T @ H_train + alpha * I, H_train.T @ Y_train_sc)
        
        H_val = np.tanh(X_val_sc @ W_in + b)
        Y_pred_val_celsius = (H_val @ beta) * sigma + mu
        Y_val_celsius = Y_val_sc * sigma + mu
        rmse_val = np.sqrt(np.mean((Y_val_celsius - Y_pred_val_celsius)**2))
        
        if rmse_val < meilleur_rmse_L:
            meilleur_rmse_L = rmse_val
            W_opt, b_opt, beta_opt = W_in, b, beta
            
    if meilleur_rmse_L < meilleur_rmse_sc:
        meilleur_rmse_sc = meilleur_rmse_L
        L_sc, W_in_sc, b_sc, beta_sc = L, W_opt, b_opt, beta_opt

print(f"--> ELM+Sin/Cos retenu : L = {L_sc} (RMSE Val = {meilleur_rmse_sc:.4f} °C)")

C_test_concat = np.concatenate((C_val[-p_final:], C_test))
X_test_sc, _ = creer_matrices_ELM_sincos(xn_test_concat_ELM, C_test_concat, p_final)

H_test_sc = np.tanh(X_test_sc @ W_in_sc + b_sc)
y_pred_ELM_sc = (H_test_sc @ beta_sc) * sigma + mu

print(f"--- Métriques ELM+Sin/Cos (Test) ---")
print(f"nRMSE : {nRMSE(y_true_test, y_pred_ELM_sc)*100:.5f} %")
print(f"NICE  : {NICE(y_true_test, y_pred_ELM_sc, pers_test):.5f}")

# ==============================================================================
# 6. GRAPHIQUES GLOBAUX
# ==============================================================================
print("\n=== ÉTAPE 6 : AFFICHAGE DES GRAPHIQUES ===")
# Les commandes plt gèrent les fenêtres d'affichage graphiques, les axes, les couleurs et les légendes.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(liste_p, erreurs_validation_AR, marker='o', color='#1f77b4')
plt.title("AR(p) : Erreur Validation")
plt.xlabel("Ordre p")
plt.ylabel("RMSE (°C)")
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(liste_L, rmse_par_L, marker='s', color='purple')
plt.title("ELM classique : Erreur Validation")
plt.xlabel("Neurones L")
plt.xscale('log') # Echelle logarithmique car L varie de 5 à 500
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

heures = 7 * 24 # On affiche seulement les 168 premières heures (1 semaine) pour la lisibilité
plt.figure(figsize=(12, 6))
plt.plot(y_true_test[:heures], label="Vraies températures", color="black", linewidth=2)
plt.plot(y_pred_AR[:heures], label=f"AR({p_final})", color="red", linestyle="--")
plt.plot(y_pred_ELM[:heures], label=f"ELM (L={meilleur_L})", color="green", linestyle="-.")
plt.plot(y_pred_ELM_sc[:heures], label=f"ELM+Sin/Cos (L={L_sc})", color="blue", linestyle=":")

plt.title("Comparaison des modèles sur la 1ère semaine de Test")
plt.xlabel("Heures")
plt.ylabel("Température (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==============================================================================
# 7. MODÈLES SUR SÉRIE DIFFÉRENCIÉE (AR et ELM)
# ==============================================================================
print("\n=== ÉTAPE 7 : MODÈLES SUR SÉRIE DIFFÉRENCIÉE ===")

# Transformer : y_t = x_t - x_{t-1}
# np.diff calcule les écarts successifs. On intègre le dernier élément du bloc précédent 
# pour éviter de perdre une ligne de données lors du calcul de la différence.
y_train = np.diff(x_train)
y_val = np.diff(np.concatenate((x_train[-1:], x_val))) 
y_test = np.diff(np.concatenate((x_val[-1:], x_test)))

# Centrage des variations
mu_y = np.mean(y_train)
yc_train = y_train - mu_y
yc_val = y_val - mu_y
yc_test = y_test - mu_y

# Normalisation pour l'ELM Différencié (pour la stabilité du tanh)
sigma_y = np.std(y_train)
yn_train = (y_train - mu_y) / sigma_y
yn_val = (y_val - mu_y) / sigma_y
yn_test = (y_test - mu_y) / sigma_y

p_diff = 24
x_prev_test = pers_test # 'pers_test' contient les x_t (la valeur de l'heure précédente)

# --- AR(24) Différencié ---
# Les calculs matriciels sont strictement identiques à l'étape 3, mais sur la série yc
X_train_yd, Y_train_yd = creer_matrices_AR(yc_train, p_diff)
w_yd, _, _, _ = np.linalg.lstsq(X_train_yd, Y_train_yd, rcond=None)

yc_test_concat = np.concatenate((yc_val[-p_diff:], yc_test))
X_test_yd, Y_test_yd_c = creer_matrices_AR(yc_test_concat, p_diff)
y_pred_AR_diff = (X_test_yd @ w_yd) + mu_y # y_hat (la prévision de la variation)

# RECOLORATION : x_hat_{t+1} = x_t + y_hat_{t+1}
# C'est ici qu'on repasse de la prévision d'une variation à la prévision de la vraie température
x_pred_AR_diff = x_prev_test + y_pred_AR_diff

print("\n--- Métriques AR(24) Différencié (Test) ---")
# L'erreur est évaluée sur les 'x' réels, d'où y_true_test vs x_pred_AR_diff
print(f"nMBE  : {nMBE(y_true_test, x_pred_AR_diff)*100:.5f} %")
print(f"nMAE  : {nMAE(y_true_test, x_pred_AR_diff)*100:.5f} %")
print(f"nRMSE : {nRMSE(y_true_test, x_pred_AR_diff)*100:.5f} %")
print(f"NICE  : {NICE(y_true_test, x_pred_AR_diff, pers_test):.5f}")

# --- ELM Différencié (Normalisé) ---
yn_val_concat_yd = np.concatenate((yn_train[-p_diff:], yn_val))
X_val_yd, Y_val_yd = creer_matrices_AR(yn_val_concat_yd, p_diff)
X_train_yn, Y_train_yn = creer_matrices_AR(yn_train, p_diff)

meilleur_rmse_y = float('inf')
L_opt_y, W_in_y, b_y, beta_y = None, None, None, None

for L in liste_L:
    meilleur_rmse_L = float('inf')
    for _ in range(nb_tirages):
        W_in = np.random.normal(0, 1, (p_diff, L))
        b = np.random.normal(0, 1, L)
        
        H_train = np.tanh(X_train_yn @ W_in + b)
        I = np.eye(L)
        beta = np.linalg.solve(H_train.T @ H_train + alpha * I, H_train.T @ Y_train_yn)
        
        H_val = np.tanh(X_val_yd @ W_in + b)
        Y_pred_val_celsius = (H_val @ beta) * sigma_y + mu_y
        Y_val_celsius = Y_val_yd * sigma_y + mu_y
        rmse_val = np.sqrt(np.mean((Y_val_celsius - Y_pred_val_celsius)**2))
        
        if rmse_val < meilleur_rmse_L:
            meilleur_rmse_L = rmse_val
            W_opt, b_opt, beta_opt = W_in, b, beta
            
    if meilleur_rmse_L < meilleur_rmse_y:
        meilleur_rmse_y = meilleur_rmse_L
        L_opt_y, W_in_y, b_y, beta_y = L, W_opt, b_opt, beta_opt

print(f"--> ELM Différencié retenu : L = {L_opt_y} (RMSE Val = {meilleur_rmse_y:.4f} °C de variation)")

yn_test_concat = np.concatenate((yn_val[-p_diff:], yn_test))
X_test_yd, _ = creer_matrices_AR(yn_test_concat, p_diff)

H_test_yd = np.tanh(X_test_yd @ W_in_y + b_y)
y_pred_ELM_diff = (H_test_yd @ beta_y) * sigma_y + mu_y

# RECOLORATION
x_pred_ELM_diff = x_prev_test + y_pred_ELM_diff

print("--- Métriques ELM Différencié (Test) ---")
print(f"nRMSE : {nRMSE(y_true_test, x_pred_ELM_diff)*100:.5f} %")
print(f"NICE  : {NICE(y_true_test, x_pred_ELM_diff, pers_test):.5f}")

# ==============================================================================
# 8. GRAPHIQUE FINAL : COMPARAISON DE TOUS LES MODÈLES
# ==============================================================================
print("\n=== ÉTAPE 8 : AFFICHAGE DU GRAPHIQUE FINAL ===")

heures = 7 * 24
plt.figure(figsize=(14, 7))
plt.plot(y_true_test[:heures], label="Vraies températures", color="black", linewidth=2.5)
plt.plot(pers_test[:heures], label="Persistance", color="gray", linestyle=":", alpha=0.7)
plt.plot(y_pred_AR[:heures], label="AR Brut", color="red", linestyle="--", alpha=0.8)
plt.plot(y_pred_ELM[:heures], label="ELM Brut", color="green", linestyle="-.", alpha=0.8)
plt.plot(x_pred_AR_diff[:heures], label="AR Différencié", color="orange", linestyle="--", linewidth=2)
plt.plot(x_pred_ELM_diff[:heures], label="ELM Différencié", color="blue", linestyle="-.", linewidth=2)

plt.title("Comparaison finale : Modèles Bruts vs Différenciés (1ère semaine de Test)")
plt.xlabel("Heures")
plt.ylabel("Température (°C)")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()