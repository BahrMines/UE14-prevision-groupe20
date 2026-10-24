import pandas as pd
import numpy as np

df = pd.read_csv('eco2mix-national-cons-def.csv', sep=';')
df=df.dropna()
print(df.head(10).loc[0:10, ['Date', 'Consommation (MW)', 'Eolien (MW)', 'Solaire (MW)', 'Nucléaire (MW)', 'Hydraulique (MW)', 'Fioul (MW)', 'Charbon (MW)', 'Gaz (MW)']])

mask= (df['Date']=='2026-08-31')
df31a = df[mask]
y=df31a[['Heure', 'Consommation (MW)']].set_index('Heure')
print(y)

