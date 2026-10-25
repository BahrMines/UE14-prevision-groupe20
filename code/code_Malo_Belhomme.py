import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
SEUIL=3 #écart maximal compris en heures accepté pour faire une interpolation de valeurs
L=[5]+[30*i for i in range(1,18)]   #valeurs de L testées
Lp=[p for p in range(1,25)]   #valeurs de p testées dans AR(p)



data=pd.read_csv('/Users/malo/Work Mines/UE14/Data_UE14_def.csv', sep=";")
mask=data['valeur brute'].notna()
data=data[mask]
data=data[['Date de début','valeur brute']]

def ecart_dates(date1,date2,seuil=SEUIL,check=False):  #seuil en nb d'heures, vérifie si on peut combler par interpolation des absences de données entre 2 dates par rapport à leur écart.   Date au format 'Y-M-D H:S:MS'
    s1,s2=date1.split(sep=' '),date2.split(sep=' ')
    s11,s21=s1[0].split('-'),s2[0].split('-')
    s12,s22=s1[1].split(':'),s2[1].split(':')
    d1,d2=s11+s12, s21+s22
    d1,d2=np.array(d1).astype(int),np.array(d2).astype(int)
    date1,date2=datetime.datetime(d1[0],d1[1],d1[2],d1[3],d1[4]),datetime.datetime(d2[0],d2[1],d2[2],d2[3],d2[4])
    deltaTime=date2-date1
    if check: return  3601<abs(deltaTime.total_seconds())<=seuil*3600
    else: return  abs(deltaTime.total_seconds())



rowPrec=data.loc[0]
for index,row in data[1:].iterrows(): #Traitement du tableau de données : interpolation ou signalement d'un vide trop grand temporellement pour être comblé

    ecart=ecart_dates(row['Date de début'],rowPrec['Date de début'])/3600

    if  1.001<ecart<=SEUIL:
        for i in range(int(ecart)-1):
            interpol = rowPrec['valeur brute'] + (i+1)*(row['valeur brute']-rowPrec['valeur brute'])/ecart
            data.loc[rowPrec.name+1+i]=['Completed', interpol]

    elif ecart >SEUIL+0.0001:
        data.loc[index-1]=['Too big gap', float(NaN)]
    rowPrec=data.loc[index]

data=data.sort_index()

l=len(data)
dataTrain=data[:int(l*8/18)]['valeur brute']
dataTest=data[int(l*8/18):-int(l*16/18)]['valeur brute']
dataValid=data[int(l*16/18):]['valeur brute']

dataTrainCentre=dataTrain-np.mean(dataTrain)
dataTestCentre=dataTest-np.mean(dataTest)
dataValidCentre=dataValid-np.mean(dataValid)



#dataTrainDecal=dataTrain.copy()
#dataTrainDecal=dataTrainDecal.iloc[1:].dropna()
#dataTrain=dataTrain.dropna()
#dataTrainDiff=dataTrain['valeur brute'].to_numpy() - np.append(dataTrainDecal['valeur brute'].to_numpy(),0)




## Apprentissage AR(p)

#Construction des tableaux X, Y et W avec les notations du cours


def XY(data,p):
    Y=[]
    X=[]
    compteur=0
    for i,val in data.items():
        if math.isnan(val): compteur=0 #si il y avait un trou trop grand dans les données qu'on n'a pas pu compléter, alors il faut reprendre à zéro les p retards utilisés pour prédire l'instant suivant
        else:
            if compteur>=p:
                Y.append([data[i]])
                X.append([data[j] for j in range(i-p,i)])
            compteur+=1
    X,Y=np.array(X),np.array(Y)
    return X,Y




dicW_ARp={p:np.zeros(p) for p in Lp}
dicY_ARp={p:[] for p in Lp}
dicX_ARp={p:[[]] for p in Lp}

for p in Lp:
    X,Y=XY(dataTrainCentre,p)
    dicY_ARp[p]=Y
    dicX_ARp[p]=X
    Xt=np.transpose(X)
    Sinv=np.linalg.inv(np.dot(Xt,X))
    inter=np.dot(Sinv,Xt)
    W=np.dot(inter,Y)
    dicW_ARp[p]=W




#idem avec la version différenciée : seule la série que l'on prédit change


def XYDiff(data,p):
    Y=[]
    X=[]
    compteur=0
    for i,val in data.loc[1:].items():
        if math.isnan(val): compteur=0
        else:
            if compteur>=p+1:
                Y.append([data[i]-data[i-1]])
                X.append([data[j]-data[j-1] for j in range(i-p,i)])
            compteur+=1
    X,Y=np.array(X),np.array(Y)
    return X,Y



dicW_ARpDiff={p:np.zeros(p) for p in Lp}
dicY_ARpDiff={p:[] for p in Lp}
dicX_ARpDiff={p:[[]] for p in Lp}

for p in Lp:
    X,Y=XYDiff(dataTrainCentre,p)
    dicY_ARpDiff[p]=Y
    dicX_ARpDiff[p]=X
    Xt=np.transpose(X)
    Sinv=np.linalg.inv(np.dot(Xt,X))
    inter=np.dot(Sinv,Xt)
    W=np.dot(inter,Y)
    dicW_ARpDiff[p]=W

## Apprentissage ELM


def paramELM(X,Y,Win,b,p,fun): #Win est tiré aléatoirement au préalable de taille p*L, ainsi que b de taille 1*L. fun est une ufunc. X,Y peuvent etre différenciés ou non
    H=fun(np.dot(X,Win)+b)
    Ht=np.transpose(H)
    S=np.dot(Ht,H)
    ridge=10**(-3)*np.eye(len(S))
    Sinv=np.linalg.inv(S+ridge)
    inter=np.dot(Sinv,Ht)
    beta=np.dot(inter,Y)
    return beta




def geneELM_fullPlageL(p,fun,diff=False):   #donne, pour une génération aléatoire de Win et b donnée, les coefficients de sortie optimisés beta
    if diff:
        X,Y=XYDiff(dataTrainCentre,p)
    else:
        X,Y=XY(dataTrainCentre,p)
    dicoBetaELM={}
    dicWin={}
    dicBiais={}
    for l in L:
        Win=np.random.uniform(-1,1,size=[p,l])
        b=np.random.uniform(-5,5,size=[1,l])
        dicWin[l]=Win
        dicBiais[l]=b
        beta=paramELM(X,Y,Win,b,p,fun)
        dicoBetaELM[l]=beta
    return dicoBetaELM ,dicWin,dicBiais


def sigmo(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


##Fonctions d'analyse statistique


def nRSME(trueValues,predictedValues,serieNonCentre):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return np.sqrt(np.square(error).sum()/N)/(np.mean(predictedValues)+np.mean(serieNonCentre))

def meanError(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return np.sqrt(np.square(error).sum()/N)

## Validation AR(p)


def ARp_predict(data):  #renvoie les valeurs predites pour data, dans un dico avec tous les p testés en clé

    dicY_ARp_predict={}
    dicY_TrueValues={}

    for p in Lp:
        Y_predict=[]
        X,Y=XY(data,p)
        dicY_TrueValues[p]=Y
        rowPrec=X[0]
        XW=np.dot(rowPrec,dicW_ARp[p])+rowPrec[0]*dicW_ARp[p][0][0]-rowPrec[-1]*dicW_ARp[p][-1][0]
        for row in X:
            XW=XW-rowPrec[0]*dicW_ARp[p][0][0]+row[-1]*dicW_ARp[p][-1][0]
            Y_predict.append(XW)
            rowPrec=row

        dicY_ARp_predict[p]=Y_predict
    return dicY_ARp_predict,dicY_TrueValues


def ARp_predictDiff(data):
    dicYDiff_TrueValues={}
    dicY_ARpDiff_predict={}
    for p in Lp:
        Y_predict=[]
        X,Y=XYDiff(data,p)
        dicYDiff_TrueValues[p]=Y
        rowPrec=X[0]
        XW=np.dot(rowPrec,dicW_ARpDiff[p])+rowPrec[0]*dicW_ARpDiff[p][0][0]-rowPrec[-1]*dicW_ARpDiff[p][-1][0]
        for row in X:
            XW=XW-rowPrec[0]*dicW_ARpDiff[p][0][0]+row[-1]*dicW_ARpDiff[p][-1][0]
            Y_predict.append(XW)
            rowPrec=row
    dicY_ARpDiff_predict[p]=Y_predict
    return  dicY_ARpDiff_predict,dicYDiff_TrueValues



##Validation ELM


def validationELM(func):
    dicY_ELM_predict_validation={} #pour chaque clé p, contient un autre dico qui pour chaque clé L détient le tableau des valeurs prédites par l'ELM
    dicY_ELMDiff_predict_validation={}

    for p in [2,5,10]:
        dicY_ELMp_predict={}
        dicY_ELMpDiff_predict={}
        dicBeta,dicWin,dicBiais=geneELM_fullPlageL(p,func)
        dicBetaDiff,dicWinDiff,dicBiaisDiff=geneELM_fullPlageL(p,func,diff=True)
        XDiff,YDiff=XYDiff(dataValidCentre,p)
        X,Y=XY(dataValidCentre,p)

        for l in L:
            Win=dicWin[l]
            biais=dicBiais[l]
            beta=dicBeta[l]
            Y_predictL=[]
            rowPrec=X[0]
            XW=np.dot(rowPrec,Win)  + rowPrec[0]*Win[0] - rowPrec[-1]*Win[-1]

            for row in X:
                XW= XW - rowPrec[0]*Win[0] + row[-1]*Win[-1]
                H=func(XW+biais)
                Y_predictL.append(np.dot(H,beta))

                rowPrec=row
            dicY_ELMp_predict[l]=Y_predictL
        dicY_ELM_predict_validation[p]=dicY_ELMp_predict


        for l in L:
            WinDiff=dicWinDiff[l]
            biaisDiff=dicBiaisDiff[l]
            betaDiff=dicBetaDiff[l]
            Y_predictLDiff=[]
            rowPrec=X[0]
            XW=np.dot(rowPrec,WinDiff)  + rowPrec[0]*WinDiff[0] - rowPrec[-1]*WinDiff[-1]

            for row in X:
                XW= XW - rowPrec[0]*WinDiff[0] + row[-1]*WinDiff[-1]
                H=func(XW+biaisDiff)
                Y_predictLDiff.append(np.dot(H,betaDiff))
                rowPrec=row
            dicY_ELMpDiff_predict[l]=Y_predictLDiff
        dicY_ELMDiff_predict_validation[p]=dicY_ELMp_predict


    return dicY_ELM_predict_validation, dicY_ELMDiff_predict_validation


##Prediction avec persistance

def persistance(data):
    X,Y=XY(data,1)
    Y_predict=[]
    for i in range(1,len(Y)):
        Y_predict.append([Y[i-1]])
    return Y_predict,Y[1:]


##

for p in Lp:
    X,Y=XY(dataValidCentre,p)
    print(len(Y),np.mean(Y),Y[0])











