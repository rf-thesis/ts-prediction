#
# original code from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#
from datetime import datetime, timedelta

import multiprocessing
from fbprophet import Prophet
import pandas as pd

from helpers.getpolygonname import getname, gettype
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
# from pandas.plotting import lag_plot, autocorrelation_plot
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from statsmodels.tsa.statespace.sarimax import SARIMAX

startdate = pd.to_datetime('2017-06-27 04:00:00')  # goes from 26-06 to 05-07
enddate = pd.to_datetime('2017-07-02 04:00:00')
filename_data = '2017_dcount15m_harmonised.csv'
filename_pols = '2017_polygonlist.csv'
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
    # transform values to float and create Series
    series = df_trimmed.drop(['ds'], axis=1)
    vals = series.values
    vals = [float(i) for i in vals]
    idx = series.index
    series_all = pd.Series(vals, idx)
    print('Series frequency: %s' % str(series_all.index.freq))
    # split into horizon (test data) and 3*horizon (train data) to make comparable to fbprophet
    split_point = len(series_all) - slices_to_predict
    series_train, series_test = series_all[0:split_point], series_all[split_point:]
    return series_all, series_train, series_test, df_trimmed


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


# calculate (S)AR-I-MA model
def calc_SARIMA(series_data, order, seasonal_order):
    print("Predicting ARIMA model %s / %s..." % (str(order), str(seasonal_order)))
    # fit model and predict
    model = SARIMAX(series_data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=0)
    pred = model_fit.get_prediction(dynamic=True, start=len(series_data) - slices_to_predict)
    # calc MAPE
    y = series_data[len(series_data) - slices_to_predict:]
    yhat = pred.predicted_mean
    # build dataframe
    df_cv = pd.concat([y, yhat], axis=1, join='inner')
    df_cv.index.names = ['ds']
    df_cv.columns = ['y', 'yhat']
    # plotSARIMA(pred, series_data)
    return df_cv


# bruteforce find optimal order
# code inspired by https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
def gridsearchSARIMA(series):
    print("Finding best ARIMA hyperparameters...")
    # Define the p, d and q parameters to take any value between 0 and 2
    p = q = range(1, 3)
    d = range(1, 2)
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
    print('Optimal parameters:', best_params)
    return best_params


# calc Auto.ARIMA
def calc_autoSARIMA(series_all):
    best_params = gridsearchSARIMA(series_all)
    df_autoarima = calc_SARIMA(series_all, best_params.param, best_params.param_seasonal)
    return df_autoarima


# create forecast model and plot, save plot under /img
def calc_fbprophet(series):
    # basic setup
    cv_horizon_amount = 24  # how far cv looks into future
    cv_horizon_unit = 'hour'  # units: sec, minute, hour, day, week, month
    # prep cols for Prophet
    model = Prophet()
    model.fit(series)
    print('model fitted.')
    from fbprophet.diagnostics import cross_validation
    cv_horizon = str(cv_horizon_amount) + ' ' + cv_horizon_unit
    df_fbprophet = cross_validation(model, horizon=cv_horizon)  # hour, day, week (singular)
    # set timestamp as index and convert to datetime
    df_fbprophet = df_fbprophet.set_index('ds')
    df_fbprophet.index = pd.to_datetime(df_fbprophet.index, infer_datetime_format=True)
    return df_fbprophet


# do stuff #
def process(polygon):
    print("Predicting polygon %s - started at %s" % (str(polygon), str(datetime.now())))
    series_all, series_train, series_test, df_all = load_data(polygon)

    # output size
    pyplot.figure(figsize=(2.5, 2))
    # calculate all models
    # calc fbprophet
    df_fbprophet = calc_fbprophet(df_all)
    fbprophet_yhat = df_fbprophet.yhat[len(df_fbprophet) - slices_to_predict:]
    mape_fbprophet = MAPE(df_fbprophet.y[len(df_fbprophet) - slices_to_predict:], fbprophet_yhat)
    # plot observed data
    df_fbprophet.y.plot(color='grey', alpha=0.7, label='observed')
    # plot predicted
    fbprophet_yhat.plot(linestyle='--', alpha=0.7, linewidth=2, label='fbprophet (MAPE: {:.2f})'.format(mape_fbprophet))

    # calc AR model
    df_AR = calc_SARIMA(series_all, (1, 0, 0), (0, 0, 0, 48))
    mape_AR = MAPE(df_AR.y, df_AR.yhat)
    df_AR.yhat.plot(linestyle='--', alpha=0.7, linewidth=2, label='AR (MAPE: {:.2f})'.format(mape_AR))

    # calc Auto.ARIMA
    mape_autoSARIMA = 0
    if autoARIMA:
        df_autoSARIMA = calc_autoSARIMA(series_all)
        mape_autoSARIMA = MAPE(df_autoSARIMA.y, df_autoSARIMA.yhat)
        df_autoSARIMA.yhat.plot(linestyle='--', alpha=0.7, linewidth=2, label='Auto-ARIMA (MAPE: {:.2f})'.format(mape_autoSARIMA))

    # format plot (http://matplotlib.org/users/customizing.html)
    # font size
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 6, 8, 10
    pyplot.rc('legend', fontsize='xx-small', loc='upper left')  # legend fontsize
    pyplot.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
    pyplot.tick_params(axis='both', which='minor', labelsize=SMALL_SIZE)
    pyplot.axes.labelsize = SMALL_SIZE
    # plot description
    pyplot.xlabel('')
    pyplot.ylabel('')
    pyplot.title('%s (%iH forecast)' % (getname(polygon), hours_to_predict), fontsize=SMALL_SIZE)
    # plot legend and set alpha
    pyplot.legend()
    pyplot.legend().get_frame().set_alpha(0.5)
    # save img
    pyplot.savefig('img/' + 'comp_pol_' + str(polygon) + '.png', bbox_inches='tight')
    #pyplot.show()
    pyplot.close()
    pyplot.clf()

    print("Finished %s at %s" % (str(polygon), str(datetime.now())))
    return {'pol': polygon,
            'pol_name': getname(polygon),
            'pol_type': gettype(polygon),
            'mape_AR': mape_AR,
            'mape_AutoARIMA': mape_autoSARIMA,
            'mape_fbprophet': mape_fbprophet}


# run stuff
from multiprocessing import Pool

polygon_list = [6, 49, 18, 5, 2, 25]  # Inner Area, Orange, Rising, Camping C, Bus/Taxi, Camping E
autoARIMA = False

if __name__ == '__main__':
    #pool = Pool()
    #results = pool.map(process, [6, 7])  # function, list TODO: change to 'polygon_list'

    results = []  # TODO: REMOVE
    results.append(process(6))
    #for i in polygon_list: results.append(process(i))

    df_results = pd.DataFrame.from_dict(results)
    df_results = df_results.set_index('pol')
    df_results.to_csv('results/2017_MAPE.csv')
