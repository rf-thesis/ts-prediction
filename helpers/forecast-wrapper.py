#
# original code from https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#
from datetime import datetime, timedelta

# import multiprocessing
# from fbprophet import Prophet
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR

startdate = pd.to_datetime('2017-06-27 04:00:00')  # goes from 26-06 to 05-07
enddate = pd.to_datetime('2017-07-02 04:00:00')
filename_data = '2017_dcount15m_harmonised.csv'
filename_pols = '2017_polygonlist.csv'
basepath = '../data/'

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
    series_train, series_test = series_all[:split_point], series_all[split_point:]
    return series_all, series_train, series_test, df_trimmed

# train autoregression
def calcAR(series_train):
    model = AR(series_train)
    model_fit = model.fit()
    print('Lag: %s' % model_fit.k_ar)
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    y = model_fit.predict(start=len(series_train), end=len(series_train)+len(series_test)-1, dynamic=True)
    return y

# do stuff
polygon = 6
series_all, series_train, series_test, df_trimmed = load_data(polygon)
calcAR(series_train)
