#
# original code from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#
from statsmodels import tsa

import dateutil
import matplotlib
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot, autocorrelation_plot
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA

startdate = dateutil.parser.parse('2017-06-27 12:00:00')  # goes from 26-06 to 05-07
enddate = dateutil.parser.parse('2017-07-02 12:00:00')
filename_data = '2017_devicecount15m.csv'
filename_pols = '2017_devicecount15m-polygons.csv'
basepath = 'data/'

polygon = 2
slices_to_predict = 100


def load_data():
    # load data
    df_orig = pd.read_csv(basepath + filename_data, squeeze=True)
    # prep cols for Prophet
    df_orig = df_orig[df_orig.polygon_id == int(polygon)]
    df_trimmed = df_orig.drop(['polygon_id'], axis=1)
    df_trimmed.columns = ['y', 'ds']
    # set date range
    df_trimmed.ds = pd.to_datetime(df_trimmed.ds, infer_datetime_format=True)
    df_trimmed = df_trimmed[(df_trimmed.ds >= startdate) & (df_trimmed.ds <= enddate)]
    df_trimmed = df_trimmed.set_index(df_trimmed.ds)
    df_trimmed = df_trimmed.drop(['ds'], axis=1)
    return df_trimmed

# calculate MAPE (timeseries evaluation metric)
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# set color
matplotlib.rcParams['axes.color_cycle'] = ['orange', 'lightblue', 'grey']

def checkAC():
    # show data
    series.plot(color="orange")
    pyplot.show()
    # check for correlation
    lag_plot(series)
    pyplot.show()
    # creates a lagged version of the dataset and calculates a correlation matrix of each column with other columns (
    # including itself)
    values = DataFrame(series.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result = dataframe.corr()
    print(result)
    # autocorrelation plot
    autocorrelation_plot(series)
    pyplot.show()
    # statsmodel AC plot
    plot_acf(series, lags=31)
    pyplot.show()

def persistancemodel():
    # build "persistance model" for baseline evaluation
    # create lagged dataset
    values = DataFrame(series.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    # split into train and test sets
    X = dataframe.values
    train, test = X[1:len(X) - slices_to_predict], X[len(X) - slices_to_predict:]
    train_X, train_y = train[:, 0], train[:, 1]
    test_X, test_y = test[:, 0], test[:, 1]
    # persistence model
    def model_persistence(x):
        return x
    # walk-forward validation
    predictions = list()
    for x in test_X:
        yhat = model_persistence(x)
        predictions.append(yhat)
    test_score = MAPE(test_y, predictions)
    print('Test MAPE: %.3f' % test_score)
    # plot predictions vs expected
    pyplot.plot(test_y)
    pyplot.plot(predictions, color='red')
    pyplot.show()

def ARIMA():
    # AR model
    # split dataset
    X = series.values
    X = [float(i) for i in X]
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = MAPE(test, predictions)
    print('Test MAPE: %.3f' % error)
    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()

########
def gridSearchOrder(series):
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)
    print(p)
    import itertools
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    print(pdq)

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = ARIMA(series, order=param)

                results = mod.fit()

                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except ValueError:
                print(ValueError)
                continue

series = load_data()
ARIMA()
