import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



##Fonctions d'analyse statistique
#prennent toutes en entrées les prédictions CENTRÉES

def nRMSE(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return np.sqrt(np.square(error).sum()/N)/abs(np.mean(predictedValues)+np.mean(dataTrain))


def nMAE(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return (np.abs(error).sum()/N)/abs(np.mean(predictedValues)+np.mean(dataTrain))

def nMBE(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return (error.sum()/N)/abs(np.mean(predictedValues)+np.mean(dataTrain))

def RSquared(trueValues,predictedValues):
    denom=np.square(predictedValues)
    errSquared=(np.square(error).sum())
    return 1-errSquared/denom

def NICE(trueValues,predictedValues):
    L=0
    err=np.abs(trueValues-predictedValues)
    N=len(predictedValues)
    for k in [1,2,3]:
        Lk=((err**k).sum()/N)**(1/k)
        L+=Lk
    return L/3