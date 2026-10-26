import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from Fonctions_statistiques import nMAE, nRMSE, nMBE, RSquared, NICE




df = pd.read_csv('data_Bahr_Chahine.csv', sep=';')
df=df.dropna() # cleaning the data by dropping the rows with missing values
#print(df.head(10).loc[0:10, ['Date', 'Consommation (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Nucléaire (MW)', 'Hydraulique (MW)', 'Fioul (MW)', 'Charbon (MW)', 'Gaz (MW)']])

mask= (df['Date']=='2026-08-31')
df31a = df[mask]
yr=df31a[['Heure', 'Consommation (MW)']].set_index('Heure') # yr : set of real values of consumption for 31/08/2026
yrr=df[(df['Date']=='2026-08-30')][['Heure', 'Consommation (MW)']].set_index('Heure') #yrr : set of real values of consumption for 30/08/2026
yrl=yr.to_numpy() # numpy array of real values of consumption for 31/08/2026
yrrl=yrr.to_numpy() # numpy array of real values of consumption for 30/08/2026


def  persistence(graph=True):

    yp=np.empty(len(yr), dtype=float) # yp : the predicted values of consumption for 31/08/2026
    yp[0]=df[(df['Date']=='2026-08-30')&(df['Heure']==yr.index[0])]["Consommation (MW)"].values[0] # the first predicted value is equal to the last registered value
    yp[1:]=yr.iloc[0:len(yr)-1,0] # applying persistence model : yp[t] = yr[t-1] 

    plt.plot(yr, label="Réel")
    plt.plot(yp, label="Prévision par persistance")
    plt.title("Consommation d'électricité pour le 31/08/2026")
    plt.xlabel("Heure",fontsize=5)
    plt.xticks(rotation=75, ticks=yr.index[::2])
    plt.ylabel("Consommation (MW)")
    plt.legend()
    if graph:
        plt.show()
    return yp   

def AR(Y,p,graph=True):
    yp=np.empty(len(Y), dtype=float) # yp : the predicted values of consumption for 31/08/2026
    X=np.column_stack([Y.to_numpy()[p-k-1:-k-1] for k in range(p)]) 
    w=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y.to_numpy()[p:])) # w : the weights of the linear regression model
    yp[p:]=np.dot(X,w)[:,0]
    #print(np.dot(X,w))

    if graph:
        plt.plot(Y[p:], label="Réel")
        plt.plot(yp[p:], label="Prévision par AR("+str(p)+")")
        plt.title("Consommation d'électricité pour le 31/08/2026")
        plt.xlabel("Heure",fontsize=5)
        plt.xticks(rotation=75, ticks=yr.index[::2])
        plt.ylabel("Consommation (MW)")
        plt.legend()
        plt.show()
    return yp


def sigmoid(x):
    return 1/(1+np.exp(-x))


def ELM(Y,p,L,a,graph=True):
    yp=np.empty(len(Y), dtype=float) # yp : the predicted values of consumption for 31/08/2026
    X=np.column_stack([Y.to_numpy()[p-k-1:-k-1] for k in range(p)]) 
    X2=(X-np.mean(X, axis=0))/np.std(X, axis=0) # X2 : the normalized input data for the sigmoid
    W=np.random.rand(p,L) # W : the weights of the hidden layer
    b=np.random.rand(L) # b : the bias of the hidden layer
    H=sigmoid(np.dot(X2,W) + b) # H : the output of the hidden layer
    beta=np.dot(np.linalg.inv(np.dot(H.T,H)+a*np.eye(L)),np.dot(H.T,Y.to_numpy()[p:])) # beta : the weights of the output layer
    yp[p:]=np.dot(H,beta)[:,0]
    print(np.dot(H,beta))

    if graph:
        plt.plot(Y[p:], label="Réel")
        plt.plot(yp[p:], label="Prévision par ELM("+str(p)+","+str(L)+")")
        plt.title("Consommation d'électricité pour le 31/08/2026")
        plt.xlabel("Heure",fontsize=5)
        plt.xticks(rotation=75, ticks=yr.index[::2])
        plt.ylabel("Consommation (MW)")
        plt.legend()
        plt.show()
    return yp

def diffELM(Y,p,L,a,graph=True):
    YL=Y.to_numpy()
    z=Y.diff().dropna()
    return ELM(z,p,L,a,graph)

def diffAR(Y,p,graph=True):
    YL=Y.to_numpy()
    z=Y.diff().dropna()
    return AR(z,p,graph)


def minpAR(graph=False):
    P=np.arange(1, 25)
    nRSMEs=np.array([nRMSE(yr, AR(yr, p,graph=False)) for p in P])
    plt.plot(P, nRSMEs.tolist(), label="nRMSE")
    plt.show()

def minpELM(graph=False):
    P=np.arange(1, 25)
    nRSMEs=np.array([nRMSE(yr, ELM(yr, p, 24, 0.01,graph=False)) for p in P])
    plt.plot(P, nRSMEs.tolist(), label="nRMSE")
    plt.show()

def minLELM(graph=False):
    Ls=np.arange(1, 300)
    nRSMEs=np.array([nRMSE(yr, ELM(yr, 2, L, 0.01,graph=False)) for L in Ls])
    plt.plot(Ls, nRSMEs.tolist(), label="nRMSE")
    plt.show()