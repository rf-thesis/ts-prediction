#
# original code from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#

data = 'slices_2017_1H.csv'
slices_to_predict = 72

import matplotlib
from pandas import Series
from matplotlib import pyplot
series = Series.from_csv(data, header=0)
print(series.head())

# set color
matplotlib.rcParams['axes.color_cycle'] = ['orange', 'red', 'grey']

# show data
series.plot(color="orange")
pyplot.show()

# check for correlation
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import lag_plot
series = Series.from_csv(data, header=0)
lag_plot(series)
pyplot.show()

# creates a lagged version of the dataset and calculates a correlation matrix of each column with other columns (
# including itself)
from pandas import Series
from pandas import DataFrame
from pandas import concat
series = Series.from_csv(data, header=0)
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)

# autocorrelation plot
from pandas import Series
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
series = Series.from_csv(data, header=0)
autocorrelation_plot(series)
pyplot.show()

# statsmodel AC plot
from pandas import Series
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series = Series.from_csv(data, header=0)
plot_acf(series, lags=31)
pyplot.show()

# build "persistance model" for baseline evaluation
from pandas import Series
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
series = Series.from_csv(data, header=0)
# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X) - slices_to_predict], X[len(X) - slices_to_predict:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]
# persistence model
def model_persistence(x):
	return x
# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()

# AR model
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
series = Series.from_csv(data, header=0)
# split dataset
X = series.values
train, test = X[1:len(X) - slices_to_predict], X[len(X) - slices_to_predict:]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()