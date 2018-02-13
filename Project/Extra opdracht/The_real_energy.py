# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:39:20 2017

@author: Rensley
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#het inlezen van data
data = pd.read_csv('2015_Energy2.csv',  sep = ';', parse_dates={'Datum en tijd(uur)':[0,1]}, index_col = 0)
geb_colommen = ["temperature"]

#de correcte kolommen vinden die temperature hebben
def data_vinden1(data, geb_colommen):
    for colom in data.columns:
        colomN = colom.replace(" ","").lower()
        for a in range(len(geb_colommen)):
            if geb_colommen[a] not in colomN:
                data.drop(colom,axis=1, inplace = True)
                
    return data

data1 = data_vinden1(data,geb_colommen)
print(data1)

#Hier maak je de kolomen dagen
data1 = data1.resample('D').mean()

### Ouwe code om tijd te groeperen 
###dataset['Tijd'] = pd.to_datetime(dataset['Tijd'], format = '%H:%M:%S').dt.hour
###sensorm = dataset.groupby([dataset['Tijd']]).mean()
###print(sensorm)
###[print(a) for a in range(0,12,2)]



#hier haal je de kolommen met de sensor 1 van sensor 0 af
cols = data1.columns
breedte = len(data1.columns)
cols2 = list()
for a in range(0,breedte,2):
    cols2.append(cols[a])
    cols2.append(cols[a+1])
    naam = ("waarden"+ str(a+1))
    data1[naam] = data1.iloc[:,a] - data1.iloc[:,(a+1)]
    cols2.append(naam)
    
data1 = data1[cols2]

print(data1)
    
#je maakt een csv bestand
data1 = data1.to_csv('Energiepy.csv')

