import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('eco2mix-national-cons-def.csv', sep=';')
df=df.dropna() # cleaning the data by dropping the rows with missing values
#print(df.head(10).loc[0:10, ['Date', 'Consommation (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Nucléaire (MW)', 'Hydraulique (MW)', 'Fioul (MW)', 'Charbon (MW)', 'Gaz (MW)']])

mask= (df['Date']=='2026-08-31')
df31a = df[mask]
yr=df31a[['Heure', 'Consommation (MW)']].set_index('Heure') # yr : set of real values of consumption for 31/08/2026
yrr=df[(df['Date']=='2026-08-30')][['Heure', 'Consommation (MW)']].set_index('Heure') #yrr : set of real values of consumption for 30/08/2026
yrl=yr.to_numpy() # numpy array of real values of consumption for 31/08/2026
yrrl=yrr.to_numpy() # numpy array of real values of consumption for 30/08/2026
print(yrr)
'''
def  persistence():

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
    plt.show()

    nicesig= np.std(yp - yrl)
    #nice2 = np.sqrt(mean_squared_error(yrl, yp))
    mae = mean_absolute_error(yrl, yp)
    mbe = np.mean(yp - yrl)

    print('NICE² :', nicesig)
    print('MAE :', mae)
    print('MBE :', mbe)
'''

def AR(p):
    yp=np.empty(len(yr), dtype=float) # yp : the predicted values of consumption for 31/08/2026
    X=np.column_stack([yrl[p-k-1:-k-1] for k in range(p)]) 
    w=np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,yrl[p:])) # w : the weights of the linear regression model
    yp[p:]=np.dot(X,w)[:,0]
    print(np.dot(X,w))

    plt.plot(yr[p:], label="Réel")
    plt.plot(yp[p:], label="Prévision par AR("+str(p)+")")
    plt.title("Consommation d'électricité pour le 31/08/2026")
    plt.xlabel("Heure",fontsize=5)
    plt.xticks(rotation=75, ticks=yr.index[::2])
    plt.ylabel("Consommation (MW)")
    plt.legend()
    plt.show()

