import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from mesures import nMAE, nRMSE, nMBE, RSquared, NICE
import sys

sys.path.insert(0, r"C:\Users\bahrc\MinesParis\UE 14 Terre et Société\Mini projet\UE14-prevision-groupe20\Fonctions")
from Fonctions_statistiques import nMAE, nRMSE, nMBE, RSquared, NICE, afficher_statistiques


#==================================================================================================
#import du tableau de données pandas



df = pd.read_csv('data_Bahr_Chahine.csv', sep=';',low_memory=False).set_index('Date et Heure')
df.index = df.index.str.replace("+02:00","",regex=False)
df.index = df.index.str.replace("T"," ", regex= False)


df=df.dropna() # on nettoies les lignes sans valeurs 
#print(df.head(10).loc[0:10, ['Date', 'Consommation (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Nucléaire (MW)', 'Hydraulique (MW)', 'Fioul (MW)', 'Charbon (MW)', 'Gaz (MW)']])



#=================================================================================================
#on filtre les données pour s'interesser à la consommation énergétique des Français sur une période précise
#ici on essaiera en particulier de prédire la consommation sur le jour du 31 aout 2026

mask= (df['Date']>='2026-08-31')
df31a = df[mask]
yr=df31a[['Consommation (MW)']]
#yrr=df[mask][['Date et Heure', 'Consommation (MW)']].set_index('Date et Heure') 
yrl=yr.to_numpy() # numpy array 
#yrrl=yrr.to_numpy() # numpy array 

#=================================================================================================
#initialisation d'une série de 20000 valeurs dont on extraiera entrainement, validation et test

X=df[['Consommation (MW)']].head(20000)

ntrain=int(len(X)*0.8)
Xtrain=X[:ntrain]

nvalid=int(len(X)*0.1)
Xvalid=X[ntrain:ntrain+nvalid]

Xtest=X[ntrain+nvalid:]

#=================================================================================================
#ici, j'implémente des fonctions qui nous serons utiles pour tous les modèles

#une fonction permettant de tracer la courbe des valeurs réelles (x) et de la prédiction (y) en fonction
#du temps pour chacun des modèles
def graphique(x,y,modele):
    plt.plot(x, label="Réel")
    plt.plot(y, label=f"Prévision par {modele}")
    plt.title("Consommation d'électricité pour le 31/08/2026")
    plt.xlabel("Heure",fontsize=5)
    plt.xticks(rotation=75, ticks=x.index[::4])
    plt.ylabel("Consommation (MW)")
    plt.legend()
    plt.show()

#une fonction donnant de nombreuses métriques relatives à un modèle de prévision et utilisant
#le fichier python Fonctions_statistiques (eventuel besoin de changer le sys.path.insert si utilisé sur un ordi autre)
def statistiques(x,y):
    print('nRMSE=',nRMSE(x,y))
    print('nMAE=',nMAE(x,y))
    print('nMBE=',nMBE(x,y))
    print('RSquared=',RSquared(x,y))
    print('NICE=',NICE(x,y,persistence(x,graph=False,stats=False)))
    print((afficher_statistiques(x,'entrée'),afficher_statistiques(y,'sortie')))


#==================================================================================================
''' 
MODELE I - PERSISTENCE 
'''
#==================================================================================================


def  persistence(Y,graph=True,stats=True):

    yp=np.empty(len(Y), dtype=float) # yp : valeurs prédites
    yp[0]=Y.iloc[0,0] # on initialise la prédiction avec la valeur instantannée
    yp[1:]=Y.iloc[0:len(Y)-1,0] # on applique le modèle de persistence : yp[t] = yr[t-1] 

    if graph:
        graphique(Y,yp,'persistence')
    if stats:
        statistiques(Y,yp)
    return yp   


#==================================================================================================
''' 
MODELE II - AUTOREGRESSION 
'''
#==================================================================================================

#à partir des données d'entrainement, on estime le poids w dans le modèle AR

def AR_woptimal(Xtrain, p):
    y = Xtrain[p:]
    X = np.column_stack([Xtrain[p-k-1:-k-1] for k in range(p)])
    w = np.linalg.solve(X.T @ X, X.T @ y)
    return w
#on obtient 
w0=AR_woptimal(Xtrain, 2)

#cette fonction renvoie le p minimisant le nRSME pour une liste de valeurs réelles d'entrée 
#le graph permet de choisir 'à la main' un p petit mais dont le nRSME, sans etre optimal est acceptable : parcimonie

def minpAR(X,graph=False):
    P=np.arange(2, 25)
    nRMSEs=np.array([nRMSE(X, AR(X, p,graph=False,stats=False)) for p in P])
    plt.plot(P, nRMSEs.tolist(), label="nRMSE")
    plt.show()
    print(np.argmin(nRMSEs.tolist())+2)

#implémentation de la fonction AR

def AR(Y,p,graph=True,stats=True):
    yp=np.empty(len(Y), dtype=float) # yp : les valeurs prédites
    X=np.column_stack([Y.to_numpy()[p-k-1:-k-1] for k in range(p)]) 
    w=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y.to_numpy()[p:])) #remarque : ici on utilise directement w = (X^T X)^-1 X^T Y car on connait mathématiquement le résultat
#                                                                        en general on ne sit peut etre pas exprimer w, on utilise alors np.linalg.lstsq (least square)
    #print(w,w.shape) # w : les poids de la régression linéaire
    yp[p:]=np.dot(X,AR_woptimal(Y,p))[:,0]
    #print(np.dot(X,w))

    if graph:
        graphique(Y[p:],yp[p:],f"AR({p})")
    if stats:
        statistiques(Y[p:],yp[p:])
    return yp

#==================================================================================================
''' 
MODELE III - AUTOREGRESSION DIFFERENCIEE
'''
#==================================================================================================

#on cherche à nouveau le p optimal pour diffAR(p)
def minpdiffAR(X,graph=False):
    P=np.arange(2, 25)
    nRMSEs=np.array([nRMSE(X, diffAR(X, p,graph=False,stats=False)) for p in P])
    if graph:
        plt.plot(P, nRMSEs.tolist(), label="nRMSE")
        plt.show()
    print(np.argmin(nRMSEs.tolist())+2)

#pour diffAR, il suffit d'utiliser AR avec la série différentiée en entrée et de retrouver en sommant la sortie 

def diffAR(Y,p,graph=True,stats=True):
    YL=Y.to_numpy()
    z=Y.diff().dropna()
    zp = AR(z,p,graph=False,stats=False)
    yp=zp+Y.shift(1).to_numpy()[:-1,0]

    if graph:
        graphique(Y[p:],yp[p:],f'diffAR({p})')
    if stats:
        statistiques(Y[p:],yp[p:])
    return yp


#==================================================================================================
''' 
MODELE IV - EXTREME LEARNING MACHINE
'''
#==================================================================================================

#comme pour un réseau de neurones lambda, on choisit une fonction d'activation qui tord la linéarité, ici une sigmoide (aurait pu etre th)
def sigmoid(x):
    return 1/(1+np.exp(-x))



#on veut optimiser les deux parametres de l'ELM, le retard p comme en AR et L la profondeur de la couche

def minpELM(X,graph=False):
    P=np.arange(2, 25)
    nRMSEs=np.array([nRMSE(X, ELM(X, p, 100, 0.01,graph=False,stats= False)) for p in P])
    if graph:
        plt.plot(P, nRMSEs.tolist(), label="nRMSE")
        plt.show()
    p_opt=np.argmin(nRMSEs)+2
    print(p_opt)
    return p_opt

def minLELM(X,graph=False):
    Ls=np.arange(1, 75)
    nRSMEs=np.array([nRMSE(X, ELM(X, 2, L, 0.01,graph=False,stats=False)) for L in Ls])
    if graph:
        plt.plot(Ls, nRSMEs.tolist(), label="nRMSE")
        plt.show()
    L_opt=np.argmin(nRSMEs.tolist())+2
    print(L_opt)
    return L_opt

#implémentation de l'ELM avec un paramètre a 

def ELM(Y,p,L,a=0.001,graph=True,stats=True):
    yp=np.empty(len(Y), dtype=float) # yp : les valeurs prédites
    X=np.column_stack([Y.to_numpy()[p-k-1:-k-1] for k in range(p)]) # X : la matrice des retards
    X2=(X-np.mean(X, axis=0))/np.std(X, axis=0) # X2 : la matrice des retards normalisée pour l'entrée dans la fonction d'activation
    W=np.random.rand(p,L) # W : les poids de la couche cachée
    b=np.random.rand(L) # b : le biais de la couche cachée
    H=sigmoid(np.dot(X2,W) + b) # H : la sortie de la couche cachée
    beta=np.dot(np.linalg.inv(np.dot(H.T,H)+a*np.eye(L)),np.dot(H.T,Y.to_numpy()[p:]))
    #print(beta)
    yp[p:]=np.dot(H,beta)[:,0]
    #print(np.dot(H,beta))

    if graph:
        graphique(Y[p:],yp[p:],f'ELM({p},{L},{a})')
    if stats:
        statistiques(Y[p:],yp[p:])
    return yp


#==================================================================================================
''' 
MODELE IV - EXTREME LEARNING MACHINE DIFFERENCIEE
'''
#==================================================================================================

#similairement à ce que l'on a vu plus tot avec l'AR

def diffELM(Y,p,L,a,graph=True,stats=True):
    YL=Y.to_numpy()
    z=Y.diff().dropna()
    zp = ELM(z,p,L,a,graph)
    yp=zp+Y.shift(1).to_numpy()[:-1,0]
    if not stats:
        print(statistiques(Y[p:],yp[p:]))
    if not graph:
        graphique(Y[p:],yp[p:],f'diffELM({p},{L},{a})')
