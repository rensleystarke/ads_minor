# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:40:13 2017

@author: Rensley
"""

from pandas import read_csv
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from matplotlib import pyplot
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
from math import sqrt

# load data
dataset = read_csv('building_clean.csv', index_col=0)
# manually specify column names
dataset.columns = ['Valve Actual Position', ' Air Flow Pressure Difference', ' Actual Air Flow', 'Temperature', ' High Voltage Switch Output', 'ActualDim', 'SwitchControlMode', 'LampEnergy', ' cTempSetpointComfort', 'cBedrijfsStatus', ' CO2 Level', 'IrtempObjectTemp', 'IrtempAmbientTemp', 'EstimatedPresence', ' LightState_Alles', ' LightDimState_Alles', 'MeasAirflow']
dataset.index.name = 'DateTime'
# mark all NA values with 0
dataset[:].fillna(0, inplace=True)
# drop the first 24 hours
#dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('fsp2.csv')

# load dataset
dataset = read_csv('fsp2.csv', header=0, index_col=0)
values = dataset.values
## specify columns to plot
groups = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
i = 1
# plot each column
#pyplot.figure()
#for group in groups:
#	pyplot.subplot(len(groups), 1, i)
#	pyplot.plot(values[:, group])
#	pyplot.title(dataset.columns[group], loc='right')
#	i += 1
#pyplot.tight_layout()
#pyplot.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('fsp2.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]], axis=1, inplace=True)
print(reframed.head())


