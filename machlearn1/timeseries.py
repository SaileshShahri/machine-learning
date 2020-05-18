"""
Lesson 01: Time Series as Supervised Learning.
Lesson 02: Load Time Series Data.
Lesson 03: Data Visualization.
Lesson 04: Persistence Forecast Model.
Lesson 05: Autoregressive Forecast Model.
Lesson 06: ARIMA Forecast Model.
Lesson 07: Hello World End-to-End Project.
"""
from pandas import read_csv
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
from math import sqrt

series = read_csv('dailybirths.csv', header=0, index_col=0)

series1 = read_csv('shampoo.csv', header=0, index_col=0)
series1.plot()
pyplot.show()

# persistence model
def model_persistence(x):
	return x

predictions = []
actual = series.values[1:]
rmse = sqrt(mean_squared_error(actual, predictions))

model = AutoReg(dataset, lags=2)
model_fit = model.fit()

prediction = model_fit.predict(start=len(dataset), end=len(dataset))

model = ARIMA(dataset, order=(0,1,0))
model_fit = model.fit()

outcome = model_fit.forecast()[0]