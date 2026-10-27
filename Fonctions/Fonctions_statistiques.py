import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



##Fonctions d'analyse statistique
#prennent toutes en entrées les prédictions CENTRÉES

def nRMSE(trueValues,predictedValues):
    trueValues, predictedValues = np.array(trueValues), np.array(predictedValues)   
    print(trueValues, predictedValues)
    error = predictedValues - trueValues #on définit ici et = x̂t − xt
    return np.sqrt(np.mean(error**2)) / np.mean(trueValues)

def nMAE(trueValues,predictedValues):
    trueValues, predictedValues = np.array(trueValues), np.array(predictedValues)
    error = predictedValues - trueValues
    return np.mean(np.abs(error)) / np.mean(trueValues)


def nMBE(trueValues,predictedValues):
    trueValues, predictedValues = np.array(trueValues), np.array(predictedValues)
    error = predictedValues - trueValues
    return np.mean(error) / np.mean(trueValues)


def RSquared(trueValues,predictedValues):
    trueValues, predictedValues = np.array(trueValues), np.array(predictedValues)
    error = predictedValues - trueValues
    num=np.sum(error**2)
    denom=np.sum((trueValues - np.mean(trueValues))**2)
    return 1-num/denom


def NICE(trueValues,predictedValues):
    trueValues = np.array(trueValues)
    predictedValues = np.array(predictedValues)
    persistanceValues = np.array(persistanceValues)
    
    error_modele= np.abs(predictedValues - trueValues)
    error_persistance = np.abs(persistanceValues - trueValues)
    
    nice=0
    for k in [1, 2, 3]:
        Lk_mod = (np.mean(err_mod**k))**(1/k)
        Lk_pers = (np.mean(err_pers**k))**(1/k)
        nice += (Lk_mod / Lk_pers)
        
    return nice / 3

def afficher_statistiques(serie, nom_serie):
    N = len(serie)
    
    # Moyenne
    m = np.mean(serie)
    
    # Écart-type (division par N implicite dans np.mean)
    sigma = np.sqrt(np.mean((serie - m)**2))
    
    # Asymétrie (Skewness)
    skewness = np.mean((serie - m)**3) / (sigma**3)
    
    # Aplatissement (Kurtosis)
    kurtosis = np.mean((serie - m)**4) / (sigma**4)
    
    print(f"-Statistiques : {nom_serie} -")
    print(f"Moyenne      : {m:.4f}")
    print(f"Écart-type   : {sigma:.4f}")
    print(f"Asymétrie    : {skewness:.4f}")
    print(f"Aplatissement: {kurtosis:.4f}\n")
    