import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
SEUIL=3 #écart maximal compris en heures accepté pour faire une interpolation de valeurs
L=[5]+[30*i for i in range(1,18)]   #valeurs de L testées
Lp=[p for p in range(1,70)]   #valeurs de p testées dans AR(p)



data=pd.read_csv('/Users/malo/WorkMines/UE14/Data_UE14_def.csv', sep=";")
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
        data.loc[index-1]=['Too big gap', float('nan')]
    rowPrec=data.loc[index]

data=data.sort_index()

l=len(data)
dataTrain=data[:int(l*8/18)]['valeur brute']
dataTest=data[int(l*8/18):int(l*16/18)]['valeur brute']
dataValid=data[int(l*16/18):]['valeur brute']

dataTrainCentre=dataTrain-np.mean(dataTrain)
dataTestCentre=dataTest-np.mean(dataTrain)
dataValidCentre=dataValid-np.mean(dataTrain)



dataDecal=data.copy()
dataDecal=dataDecal.dropna().iloc[1:]
data=data.dropna()
dataDiff=data['valeur brute'].to_numpy() - np.append(dataDecal['valeur brute'].to_numpy(),0)




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
    YRebuild=[]
    compteur=0
    for i,val in data.loc[1:].items():
        if math.isnan(val): compteur=0
        else:
            if compteur>=p+1:
                Y.append([data[i]-data[i-1]])
                X.append([data[j]-data[j-1] for j in range(i-p,i)])
                YRebuild.append([data[i-1]])
            compteur+=1
    X,Y=np.array(X),np.array(Y)

    return X,Y,YRebuild



dicW_ARpDiff={p:np.zeros(p) for p in Lp}
dicY_ARpDiff={p:[] for p in Lp}
dicX_ARpDiff={p:[[]] for p in Lp}

for p in Lp:
    X,Y,dicRebuild=XYDiff(dataTrainCentre,p)
    dicY_ARpDiff[p]=Y
    dicX_ARpDiff[p]=X
    Xt=np.transpose(X)
    Sinv=np.linalg.inv(np.dot(Xt,X))
    inter=np.dot(Sinv,Xt)
    W=np.dot(inter,Y)
    dicW_ARpDiff[p]=W

## Apprentissage ELM


def paramELM(X,Y,Win,b,p,fun): #Win est tiré aléatoirement au préalable de taille p*L, ainsi que b de taille 1*L. fun est une ufunc. X,Y peuvent etre différenciés ou non
    if func==sigmo: norm=10
    else: norm =3
    H=fun(np.dot(X,Win)/5+b)
    Ht=np.transpose(H)
    S=np.dot(Ht,H)
    ridge=10**(-3)*np.eye(len(S))
    Sinv=np.linalg.inv(S+ridge)
    inter=np.dot(Sinv,Ht)
    beta=np.dot(inter,Y)
    return beta




def geneELM_fullPlageL(p,fun,diff=False):   #donne, pour une génération aléatoire de Win et b donnée, les coefficients de sortie optimisés beta
    if diff:
        X,Y,YRebuild=XYDiff(dataTrainCentre,p)
    else:
        X,Y=XY(dataTrainCentre,p)
    dicoBetaELM={}
    dicWin={}
    dicBiais={}
    for l in L:
        Win=np.random.uniform(-1,1,size=[p,l])
        b=np.random.uniform(-1,1,size=[1,l])
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
#prennent toutes en entrées les prédictions CENTRÉES

def nRMSE(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return np.sqrt(np.square(error).sum()/N)/abs(np.mean(trueValues)+np.mean(dataTrain))


def nMAE(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return (np.abs(error).sum()/N)/abs(np.mean(trueValues)+np.mean(dataTrain))

def nMBE(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    return (error.sum()/N)/abs(np.mean(trueValues)+np.mean(dataTrain))

def RSquared(trueValues,predictedValues):
    N=len(predictedValues)
    error=np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N)
    denom=np.square(trueValues - np.mean(trueValues)).sum()
    errSquared=(np.square(error).sum())
    return 1-errSquared/denom



def NICE_sigma(trueValues,predictedValues,Data):
    L=0
    N=len(predictedValues)
    error=np.abs(np.array(predictedValues).reshape(N)-np.array(trueValues).reshape(N))
    Ypersistance,YTruePersistance=persistance(Data)
    Npersistance=len(Ypersistance)
    errPersistance=np.abs(np.array(Ypersistance).reshape(Npersistance)-np.array(YTruePersistance).reshape(Npersistance))

    for k in [1,2,3]:
        Lk=((error**k).sum()/N)**(1/k)
        Lkpersistance=((errPersistance**k).sum()/Npersistance)**(1/k)
        L+=Lk/Lkpersistance
    return L/3


## Validation AR(p)


def ARp_predict(data):  #renvoie les valeurs predites pour data, dans un dico avec tous les p testés en clé

    dicY_ARp_predict={}
    dicY_TrueValues={}

    for p in Lp:
        Y_predict=[]
        X,Y=XY(data,p)
        dicY_TrueValues[p]=Y
        #rowPrec=X[0]
        #XW=np.dot(rowPrec,dicW_ARp[p])+rowPrec[0]*dicW_ARp[p][0][0]-rowPrec[-1]*dicW_ARp[p][-1][0]
        #for row in X:
        #    XW=XW-rowPrec[0]*dicW_ARp[p][0][0]+row[-1]*dicW_ARp[p][-1][0]
        #    Y_predict.append(XW)
        #    rowPrec=row
        for row in X:
            Y_predict.append(np.dot(row,dicW_ARp[p]))

        dicY_ARp_predict[p]=Y_predict
    return dicY_ARp_predict,dicY_TrueValues


def ARp_predictDiff(data):
    dicYDiff_TrueValues={}
    dicY_ARpDiff_predict={}
    dicYRebuild={}
    for p in Lp:
        Y_predict=[]
        X,Y,YRebuild=XYDiff(data,p)
        dicYRebuild[p]=YRebuild
        #rowPrec=X[0]
        #XW=np.dot(rowPrec,dicW_ARpDiff[p])+rowPrec[0]*dicW_ARpDiff[p][0][0]-rowPrec[-1]*dicW_ARpDiff[p][-1][0]
        for row in X:
            #XW=XW-rowPrec[0]*dicW_ARpDiff[p][0][0]+row[-1]*dicW_ARpDiff[p][-1][0]
            #rowPrec=row
            XW=np.dot(row,dicW_ARpDiff[p])
            Y_predict.append(XW)

        dicYDiff_TrueValues[p]=Y
        dicY_ARpDiff_predict[p]=Y_predict
    return  dicY_ARpDiff_predict,dicYDiff_TrueValues, dicYRebuild



##Validation ELM


def ELM_predict(data,func,funcdiff):
    dicY_ELM_predict={} #pour chaque clé p, contient un autre dico qui pour chaque clé L détient le tableau des valeurs prédites par l'ELM
    dicY_ELMDiff_predict={}
    dicY_True={}
    dicYDiff_True={}
    dicYRebuild={}
    for p in [2,5,10]:
        dicY_ELMp_predict={}
        dicY_ELMpDiff_predict={}
        dicBeta,dicWin,dicBiais=geneELM_fullPlageL(p,func)
        dicBetaDiff,dicWinDiff,dicBiaisDiff=geneELM_fullPlageL(p,funcdiff,diff=True)
        XDiff,YDiff,YRebuild=XYDiff(data,p)
        dicYRebuild[p]=YRebuild[:-1]
        X,Y=XY(data,p)
        dicY_True[p]=Y[:-1]
        dicYDiff_True[p]=YDiff[:-1]
        for l in L:
            Win=dicWin[l]
            biais=dicBiais[l]
            beta=dicBeta[l]
            Y_predictL=[]
            #rowPrec=X[0]
            #XW=np.dot(rowPrec,Win)  + rowPrec[0]*Win[0] - rowPrec[-1]*Win[-1]

            for row in X:
                #XW= XW - rowPrec[0]*Win[0] + row[-1]*Win[-1]
                XW=np.dot(row,Win)
                H=func(XW+biais)
                Y_predictL.append(np.dot(H,beta))

                #rowPrec=row
            dicY_ELMp_predict[l]=Y_predictL[1:]
        dicY_ELM_predict[p]=dicY_ELMp_predict


        for l in L:
            WinDiff=dicWinDiff[l]
            biaisDiff=dicBiaisDiff[l]
            betaDiff=dicBetaDiff[l]
            Y_predictLDiff=[]
            #rowPrec=XDiff[0]
            #XW=np.dot(rowPrec,WinDiff)  + rowPrec[0]*WinDiff[0] - rowPrec[-1]*WinDiff[-1]

            for row in XDiff:
                #XW= XW - rowPrec[0]*WinDiff[0] + row[-1]*WinDiff[-1]
                XW=np.dot(row,WinDiff)
                H=func(XW+biaisDiff)
                Y_predictLDiff.append(np.dot(H,betaDiff))
                #rowPrec=row
            dicY_ELMpDiff_predict[l]=Y_predictLDiff[1:]
        dicY_ELMDiff_predict[p]=dicY_ELMpDiff_predict
    return (dicY_ELM_predict,dicY_True), (dicY_ELMDiff_predict, dicYDiff_True, dicYRebuild)


##Prediction avec persistance

def persistance(data):
    X,Y=XY(data,1)
    Y_predict=[]
    for i in range(1,len(Y)):
        Y_predict.append([Y[i-1]])
    return Y_predict,Y[1:]


##Fonctions validation



def trace_valid_nRMSE_ELM(data):
    tabFun=[sigmo,tanh]
    tabFunString=['sigmo','tanh']
    fig,axes=plt.subplots(2,2)
    for i,func in enumerate(tabFun):


        (dicY_pred,dicY_True),(dicYDiff_pred,dicYDiff_True,dicYRebuild)=ELM_predict(data,func,func)

        for j,diff in enumerate(['non diff','diff']):
            if diff=='diff': dicY_predict, dicY_TrueValues=dicYDiff_pred,dicYDiff_True
            else: dicY_predict, dicY_TrueValues=dicY_pred,dicY_True
            ax=axes[i,j]
            for (p,dicoYPred),(d,dicoYTrue) in zip(dicY_predict.items(), dicY_TrueValues.items()):
                nRMSEtab=[]
                for l in L:
                    Ypred=dicoYPred[l]
                    Y=dicoYTrue
                    if diff=='diff':
                        YRebuild=dicYRebuild[p]
                        Ypred=[Ypred[i][0]+YRebuild[i][0] for i in range(len(YRebuild)) ]
                        Y=[Y[i][0]+YRebuild[i][0] for i in range(len(YRebuild)) ]
                    nRMSEtab.append(nRMSE(Y,Ypred)*100)
                ax.plot(L,nRMSEtab,label=f"{p} retards en entrée")
            ax.set_xlabel("Valeurs de L, nombre de neurones")
            ax.set_ylabel('nRMSE en %')
            ax.set_title(f"Série {diff} avec la fonction d'activation {tabFunString[i]}")
            ax.legend()
    plt.suptitle('nRMSE des séries différenciées ou non, avec tanh ou sigmoïde en activation, avec la méthode ELM')
    plt.show()

def trace_valid_nRMSE_ARp(data):
    dicY_ARp_predict,dicY_TrueValues=ARp_predict(data)
    fig,axes=plt.subplots(1,2)
    (dicY_pred,dicY_True),(dicYDiff_pred,dicYDiff_True,dicYRebuild)=ARp_predict(data),ARp_predictDiff(data)
    for j,diff in enumerate(['non diff','diff']):
        if diff=='diff': dicY_predict, dicY_TrueValues=dicYDiff_pred,dicYDiff_True
        else: dicY_predict, dicY_TrueValues=dicY_pred,dicY_True
        ax=axes[j]
        nRMSEtab=[]
        for p in dicY_predict:
            Ypred=dicY_predict[p]
            Y=dicY_TrueValues[p]
            if diff=='diff':
                YRebuild=dicYRebuild[p]
                Ypred=[Ypred[i][0]+YRebuild[i][0] for i in range(len(YRebuild)) ]
                Y=[Y[i][0]+YRebuild[i][0] for i in range(len(YRebuild)) ]
            nRMSEtab.append(nRMSE(Y[:-1],Ypred[1:])*100)
        ax.plot(Lp,nRMSEtab)
        ax.set_xlabel('nombre de retards p')
        ax.set_ylabel('nRMSE en %')
        ax.set_title(f"Série {diff}")
    plt.suptitle("nRMSE des séries différenciées ou non, avec la méthode AR(p)" )
    plt.show()

##Tracés validation

trace_valid_nRMSE_ARp(dataValidCentre)
trace_valid_nRMSE_ELM(dataValidCentre)

##Tests des modèles

p_ARp=2
p_ARpDiff=1
p_ELM=10
p_ELMDiff=5
L_ELM=5
L_ELMDiff=5
func=sigmo
funcDiff=tanh

def Rebuild(YDiff,YRebuild):
    return [YDiff[i][0]+YRebuild[i][0] for i in range(len(YRebuild))]


(d1,d2),(d3,d4,d5)=ELM_predict(dataTestCentre,func,funcDiff)
(YPredELM,YTrueELM),(YDiffPredELM,YDiffTrueELM,YRebuildELM)=(d1[p_ELM][L_ELM],d2[p_ELM]),(d3[p_ELMDiff][L_ELMDiff], d4[p_ELMDiff], d5[p_ELMDiff])
YDiffPredELM,YDiffTrueELM=Rebuild(YDiffPredELM,YRebuildELM)[1:], Rebuild(YDiffTrueELM,YRebuildELM)[:-1]


d1,d2=ARp_predict(dataTestCentre)
YPredARp,YTrueARp=d1[p_ARp][1:],d2[p_ARp][:-1]

d1,d2,d3=ARp_predictDiff(dataTestCentre)
YRebuildARp=d3[p_ARpDiff]
YPredARpDiff,YTrueARpDiff=Rebuild(d1[p_ARpDiff],YRebuildARp)[1:],Rebuild(d2[p_ARpDiff],YRebuildARp)[:-1]


YPredPersistance,YTruePersistance=persistance(dataTestCentre)

print(f"Le nRMSE des différents modèles est: {nRMSE(YTrueELM,YPredELM)*100}% pour l'ELM, {nRMSE(YDiffTrueELM,YDiffPredELM)*100}% pour l'ELM différencié, {nRMSE(YTrueARp,YPredARp)*100}% pour l'AR, et {nRMSE(YTrueARpDiff,YPredARpDiff)*100}% pour l'AR différencié. Enfin on a un nRMSE de {nRMSE(YTruePersistance,YPredPersistance)*100}% pour le modèle de référence de persistance.")


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


afficher_statistiques(data['valeur brute'],"pollution de l'air")
afficher_statistiques(dataDiff,"pollution de l'air différencié")


LYpred=[ YPredARp,YPredELM,YPredPersistance,YPredARpDiff, YDiffPredELM]
LYtrue=[ YTrueARp,YTrueELM, YTruePersistance,YTrueARpDiff,YDiffTrueELM]
Ltitle=['AR','ELM','Persistance','AR Diff', 'ELM Diff']
LmetricsName=['nMAE','nMBE','RSquared','NICE_sigma']
for i,metrique in enumerate([nMAE,nMBE,RSquared,NICE_sigma]):
    for k in range(5):
        if i!=3:
            print(f'la métrique {LmetricsName[i]} vaut {metrique(LYtrue[k], LYpred[k])}% pour le modèle {Ltitle[k]} ')
        else: print(f'la métrique {LmetricsName[i]} vaut {metrique(LYtrue[k], LYpred[k],dataTestCentre)} pour le modèle {Ltitle[k]} ')




##Graphes de visualisation


#tracé des différentes prédictions sur une période de N heures
N=200
LYpred=[ YPredARp,YPredELM,YPredPersistance,YPredARpDiff, YDiffPredELM]
LYtrue=[ YTrueARp,YTrueELM, YTruePersistance,YTrueARpDiff,YDiffTrueELM]
Ltitle=['AR','ELM','Persistance','AR Diff', 'ELM Diff']
tabt=[i for i in range(N)]
fig,axes=plt.subplots(2,3)
axes[1,2].set_visible(False)
#axes[0][1].set_position([0.24,0.125,0.228,0.343])

for k in range(5):
    i=k//3
    j=k%3
    axes[i,j].plot(tabt, np.asarray(LYpred[k]).reshape(-1)[1000-N:1000], ls='-', color='red', label='Prédiction')
    axes[i,j].plot(tabt, np.asarray(LYtrue[k]).reshape(-1)[1000-N:1000] ,color='black', label='Vraies valeures')
    axes[i,j].set_ylabel('Pollution en microgramme/m^3 de PM10')
    axes[i,j].set_xlabel('temps en heures')
    axes[i,j].set_title(f'Prédiction avec {Ltitle[k]}')
    axes[i,j].get_legend()
plt.legend()
plt.show()


tabnRMSE=[nRMSE(YTruePersistance,YPredPersistance)*100, nRMSE(YTrueELM,YPredELM)*100, nRMSE(YDiffTrueELM,YDiffPredELM)*100, nRMSE(YTrueARp,YPredARp)*100, nRMSE(YTrueARpDiff,YPredARpDiff)*100, ]
tabModels=['Persistance', 'ELM', 'ELM DIff', 'ARp', 'ARp Diff']
plt.figure()
plt.bar(tabModels, tabnRMSE)
plt.ylabel('nRMSE en %')
plt.title('Performance de chaque modèle')
plt.show()









