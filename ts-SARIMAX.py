#
# original code from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#
from datetime import datetime, timedelta

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
from statsmodels.tsa.statespace.sarimax import SARIMAX

tolerance = timedelta(minutes=5)

startdate = pd.to_datetime('2017-06-27 04:00:00') - tolerance  # goes from 26-06 to 05-07
enddate = pd.to_datetime('2017-07-02 04:00:00') + tolerance
filename_data = '2017_devicecount15m_harmonised.csv'
filename_pols = '2017_devicecount15m-polygons.csv'
basepath = 'data/'

hours_to_predict = 24
slices_to_predict = hours_to_predict * 4


def load_data(polygon):
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
    # harmonise timestamp to have a regular 15m frequency
    # df_trimmed.index = pd.date_range(start=startdate, end=enddate, freq='15min')
    # transform values to float and create Series
    series = df_trimmed.drop(['ds'], axis=1)
    vals = series.values
    vals = [float(i) for i in vals]
    idx = series.index
    series_all = pd.Series(vals, idx)
    # series_all = series_all.asfreq('15min')
    print('Series frequency: %s' % str(series_all.index.freq))
    # split into horizon (test data) and 3*horizon (train data) to make comparable to fbprophet
    split_point = len(series_all) - slices_to_predict
    series_train, series_test = series_all[0:split_point], series_all[split_point:]
    return series_all, series_train, series_test


# calculate MAPE (timeseries evaluation metric)
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def checkAC():
    # show data
    series_all.plot(color="orange")
    pyplot.show()
    # check for correlation
    lag_plot(series_all)
    pyplot.show()
    # creates a lagged version of the dataset and calculates a correlation matrix of each column with other columns (
    # including itself)
    values = DataFrame(series_all.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    result = dataframe.corr()
    print(result)
    # autocorrelation plot
    autocorrelation_plot(series_all)
    pyplot.show()
    # statsmodel AC plot
    plot_acf(series_all, lags=31)
    pyplot.show()


def persistancemodel():
    # build "persistance model" for baseline evaluation
    # create lagged dataset
    values = DataFrame(series_all.values)
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
    pyplot.clf()


def calcARIMA_long():
    # AR model
    # split dataset
    X = series_all.values
    X = [float(i) for i in X]
    size = int(len(X) - slices_to_predict)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(0, 1, 0))
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
    pyplot.clf()


# xxx
def calc_ARIMA(series_all, order, seasonal_order):
    print("Predicting ARIMA model %s / %s..." % (str(order), str(seasonal_order)))
    # fit model and predict
    model = SARIMAX(series_all, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=0)
    pred = model_fit.get_prediction(dynamic=True, start=len(series_all) - slices_to_predict)
    # calc MAPE
    y_true = series_all[len(series_all) - slices_to_predict:]
    y_pred = pred.predicted_mean
    # build dataframe
    df_cv = pd.concat([y_true, y_pred], axis=1, join='inner')
    df_cv.index.names = ['ds']
    df_cv.columns = ['y', 'y_hat']
    print(df_cv.head(n=10))
    print('MAPE: % .2f' % MAPE(y_true, y_pred))
    plot(pred, series_all)
    return df_cv

def plot(pred, series_all):
    # plot graph
    pred_ci = pred.conf_int()
    ax = series_all.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    pyplot.legend()
    pyplot.show()


# bruteforce find optimal order
def gridSearchOrder(series):
    print("Finding best ARIMA hyperparameters...")
    # Define the p, d and q parameters to take any value between 0 and 2
    p = q = range(0, 2)
    d = range(0, 2)
    import itertools
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 48) for x in list(itertools.product(p, d, q))]

    all_params = pd.DataFrame(columns=['param', 'param_seasonal', 'aic'])
    idx = 0
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(series,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = mod.fit(disp=0)
                all_params.loc[idx] = [param, param_seasonal, results.aic]
                print(all_params.loc[idx])
                idx += 1
            except ValueError:
                print(ValueError)
                continue

    best_aic_idx = all_params.aic.idxmin()
    best_params = all_params.loc[best_aic_idx]
    print(all_params)
    print('optimal parameters:', best_params)
    return best_params

# calc Auto.ARIMA
def calc_autoARIMA(series_all):
    best_params = gridSearchOrder(series_all)
    calc_ARIMA(series_all, best_params.param, best_params.param_seasonal)  # “frequency” argument is the number of observations per season

# do stuff
polygon = 38
print("Predicting polygon %s - started at %s" % (str(polygon), str(datetime.now())))
series_all, series_train, series_test = load_data(polygon)
# calc Auto.ARIMA
calc_autoARIMA()
# calc AR model
calc_ARIMA(series_all, (1, 0, 0), (0, 0, 0, 48))
print("Finished at %s" % (str(polygon), str(datetime.now())))
