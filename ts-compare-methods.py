#
# original code from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#
from datetime import datetime, timedelta

import multiprocessing
from fbprophet import Prophet
import pandas as pd
from statsmodels.tsa.ar_model import AR

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
global_freq = None


def load_data(polygon):
    # load data
    df_orig = pd.read_csv(basepath + filename_data, squeeze=True)
    # prep cols for Prophet
    df_orig = df_orig[df_orig.polygon_id == int(polygon)]
    df_fbprophet = df_orig.drop(['polygon_id'], axis=1)
    df_fbprophet.columns = ['y', 'ds']
    # set date range
    df_fbprophet.ds = pd.to_datetime(df_fbprophet.ds, infer_datetime_format=True)
    df_fbprophet = df_fbprophet[(df_fbprophet.ds >= startdate) & (df_fbprophet.ds <= enddate)]
    # reindex for Series conversion
    # set new index
    df_reindexed = df_fbprophet.set_index(df_fbprophet.ds)
    # remove duplicate indices
    df_reindexed = df_reindexed[~df_reindexed.index.duplicated(keep='first')] # print(df_fbprophet[df_fbprophet.index.duplicated()])
    # rename index
    df_reindexed = df_reindexed.drop(['ds'], axis=1)
    df_reindexed.index.rename('ds', inplace=True)
    # set frequency
    df_reindexed = df_reindexed.asfreq(freq='15min')
    global_freq = df_reindexed.index.freq
    print(global_freq)
    # transform values to float and create Series
    vals = df_reindexed.values
    vals = [float(i) for i in vals]
    idx = df_reindexed.index
    series_all = pd.Series(vals, idx)
    series_all.asfreq(freq='15min')
    # split into horizon (test data) and 3*horizon (train data) to make comparable to fbprophet
    split_point = len(series_all) - slices_to_predict
    series_train, series_test = series_all[:split_point], series_all[split_point:]
    return series_all, series_train, series_test, df_fbprophet


# calculate MAPE (timeseries evaluation metric)
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def SMAPE(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)


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


# calculate AR Model
def calc_AR(series_train, series_test):
    model = AR(series_train)
    model_fit = model.fit()
    print('Optimal # of lags: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    y = series_test
    yhat = model_fit.predict(start=len(series_train), end=len(series_train)+len(series_test)-1, dynamic=True)
    return y, yhat


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
    # output size (must be up here)
    pyplot.figure(figsize=(3, 2))

    # calculate all models
    # calc AR model
    AR_y, AR_yhat = calc_AR(series_train, series_test)
    smape_AR = SMAPE(AR_y, AR_yhat)
    AR_yhat.plot(color='green', linestyle='--', alpha=0.65, linewidth=1.5, label='AR (SMAPE: {:.2f})'.format(smape_AR))

    # calc fbprophet
    df_fbprophet = calc_fbprophet(df_all)
    fbprophet_yhat = df_fbprophet.yhat[len(df_fbprophet) - slices_to_predict:]
    smape_fbprophet = SMAPE(df_fbprophet.y[len(df_fbprophet) - slices_to_predict:], fbprophet_yhat)
    df_fbprophet.y.plot(color='grey', alpha=0.65, label='observed')     # plot observed data
    fbprophet_yhat.plot(color='blue', linestyle='--', alpha=0.65, linewidth=1.5, label='fbprophet (SMAPE: {:.2f})'.format(smape_fbprophet))       # plot predicted

    # calc Auto.ARIMA
    smape_autoSARIMA = 0
    if autoARIMA:
        df_autoSARIMA = calc_autoSARIMA(series_all)
        smape_autoSARIMA = SMAPE(df_autoSARIMA.y, df_autoSARIMA.yhat)
        df_autoSARIMA.yhat.plot(color='red', linestyle='--', alpha=0.65, linewidth=1.5, label='Auto-ARIMA (SMAPE: {:.2f})'.format(smape_autoSARIMA))

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
    #pyplot.legend().get_frame().set_alpha(0.5)
    # save img
    pyplot.savefig('img/compare/' + 'comp_pol_' + str(polygon) + '.png', bbox_inches='tight')
    #pyplot.show()
    pyplot.close()
    pyplot.clf()

    print("Finished %s at %s" % (str(polygon), str(datetime.now())))
    return {'pol': polygon,
            'pol_name': getname(polygon),
            'pol_type': gettype(polygon),
            'smape_AR': smape_AR,
            'smape_AutoARIMA': smape_autoSARIMA,
            'smape_fbprophet': smape_fbprophet}


# run stuff
from multiprocessing import Pool

#will not work for some polys
#df_polygons = pd.read_csv('data/2017_polygoninfo_filtered.csv', usecols=["ogr_fid"], nrows=None)
#polygon_list = df_polygons.values.astype(int).flatten()
polygon_list = 49
polygon_list = [6, 49, 18, 5, 25, 11, 16, 10]  # Inner Area, Orange, Rising, Camping C + E, Bridge, Tradezone, Street City
autoARIMA = True

if __name__ == '__main__':
    if isinstance(polygon_list, int):
        results = []
        process(polygon_list)
    else:
        pool = Pool()
        results = pool.map(process, polygon_list)
    df_results = pd.DataFrame.from_dict(results)
    df_results = df_results.set_index('pol')
    df_results.to_csv('results/2017_SMAPE.csv')
