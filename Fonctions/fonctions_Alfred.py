import numpy as np


def persistance(serie):
    # serie.shift(1) décale les valeurs d’une ligne vers le bas en conservant les dates.
    return serie.shift(1)

def statistiques(serie):
    moyenne = np.mean(serie)
    ecart_type = np.std(serie, ddof=0)
    valeurs_centrees = serie - moyenne
    asymetrie = np.mean(valeurs_centrees ** 3) / ecart_type ** 3
    kurtosis = np.mean(valeurs_centrees ** 4) / ecart_type ** 4

    return {
        "moyenne": moyenne,
        "ecart_type": ecart_type,
        "asymetrie": asymetrie,
        "kurtosis": kurtosis}

def prevision_ar(apprentissage, historique, p, positif=True):
    # Centrer les données avec la moyenne d'apprentissage uniquement
    # La moyenne est calculée sur l’apprentissage uniquement
    moyenne =apprentissage.mean()
    train =apprentissage.to_numpy()- moyenne
    # Chaque ligne contient p prix passés, du plus récent au plus ancien
    X = []
    y = []
    for t in range(p, len(train)):
        # Avec cette ligne on récupère les p valeurs précédant la position t
        X.append(train[t-p:t][::-1])
        y.append(train[t])

    X = np.array(X)
    y = np.array(y)

    # On apprend les coefficients par moindres carrés
    poids =np.linalg.lstsq(X, y, rcond=None)[0]

    # On prévoit chaque heure de validation avec les observations précédentes
    valeurs =historique.to_numpy() - moyenne
    previsions= []

    # On parcourt uniquement les positions de validation :
    for t in range(len(train), len(valeurs)):
        passe =valeurs[t-p:t][::-1]
        prediction =moyenne + np.dot(poids, passe)
        if positif:
            prediction =max(0, prediction)
        previsions.append(prediction)

    return np.array(previsions)

def nmbe(mesures, previsions):
    mesures = np.asarray(mesures)
    previsions = np.asarray(previsions)
    erreurs = previsions - mesures
    return 100 * np.mean(erreurs) / np.mean(mesures)


def nmae(mesures, previsions):
    mesures = np.asarray(mesures)
    previsions = np.asarray(previsions)
    erreurs = previsions - mesures
    return 100 * np.mean(np.abs(erreurs)) / np.mean(mesures)


def nrmse(mesures, previsions):
    mesures = np.asarray(mesures)
    previsions = np.asarray(previsions)
    erreurs = previsions - mesures
    return 100 * np.sqrt(np.mean(erreurs ** 2)) / np.mean(mesures)


def r2(mesures, previsions):
    mesures = np.asarray(mesures)
    previsions = np.asarray(previsions)
    erreurs = previsions - mesures
    return 1 - np.sum(erreurs ** 2) / np.sum(
        (mesures - np.mean(mesures)) ** 2)


def nice_sigma(mesures, previsions, previsions_persistance):
    mesures = np.asarray(mesures)
    previsions = np.asarray(previsions)
    previsions_persistance = np.asarray(previsions_persistance)

    erreurs =previsions- mesures
    erreurs_persistance= previsions_persistance - mesures

    rapports= []

    for k in [1, 2, 3]:
        erreur_modele =np.mean(np.abs(erreurs) ** k) ** (1 / k)
        erreur_reference =np.mean(np.abs(erreurs_persistance) ** k)** (1 / k)

        rapports.append(erreur_modele / erreur_reference)

    return np.mean(rapports)

def bilan_erreurs(mesures, previsions, previsions_persistance):
    return {
        "nMBE (%)": nmbe(mesures, previsions),
        "nMAE (%)": nmae(mesures, previsions),
        "nRMSE (%)": nrmse(mesures, previsions),
        "R2": r2(mesures, previsions),
        "NICE_sigma": nice_sigma(
            mesures, previsions, previsions_persistance)}

def prevision_elm(apprentissage, historique, p, L, graine, positif=True):
    # On normaliser avec la moyenne et l'écart-type d'apprentissage uniquement
    moyenne =apprentissage.mean()
    ecart_type = np.std(apprentissage, ddof=0)
    valeurs =(historique.to_numpy()- moyenne)/ecart_type
    n = len(apprentissage)

    # On fait pareil que pour l'AR avec p observations précédentes et la récente en premier.
    X_train =[]
    for t in range(p, n):
        X_train.append(valeurs[t-p:t][::-1])
    X_train =np.array(X_train)
    y_train =valeurs[p:n]

    X_validation =[]
    for t in range(n,len(valeurs)):
        X_validation.append(valeurs[t-p:t][::-1])
    X_validation =np.array(X_validation)

    # On fixe les poids d'entrée et les biais aléatoires
    hasard =np.random.default_rng(graine)
    poids_entree= hasard.uniform(-1, 1, size=(p, L))
    biais = hasard.uniform(-1, 1, size=L)

    # On transforme les entrées avec les L neurones.
    H_train =np.tanh(X_train @ poids_entree + biais)
    H_validation = np.tanh(X_validation @ poids_entree + biais)

    # On apprend seulement les poids de sortie
    alpha= 0.001
    poids_sortie= np.linalg.solve(
        H_train.T @ H_train + alpha * np.eye(L),
        H_train.T @ y_train)

    # Finalement on revient aux unités de la série d'origine.
    previsions = moyenne +ecart_type*(H_validation @ poids_sortie)
    if positif:
        previsions = np.maximum(0, previsions)

    return previsions