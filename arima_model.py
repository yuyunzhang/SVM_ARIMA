# -*- coding: utf-8 -*-
"""
Created on Wed May 17 22:06:39 2017

@author: biank
@reference: http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
"""

from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
 
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')
 
series = read_csv('C:/Users/biank/Desktop/Machine Learning Finance/Final project/PM.csv',sep=',',usecols=[0,4], parse_dates=[0], header=0, index_col=0, squeeze=True, date_parser = parser)
"""
model = ARIMA(series,order=(0,1,0))
model_fit = model.fit()

print(model_fit.summary())
#plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
"""

X = series.values
#size = int(len(X) * 0.995)
size = 50
#train, test = X[0:size], X[size:len(X)]
train = X[0:10]
test = X[10:size]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test, color='blue')
pyplot.plot(predictions, color='red')
pyplot.show()















