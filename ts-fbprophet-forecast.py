from collections import defaultdict
from itertools import chain

import dateutil
import numpy
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Manager
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# setup data
slices_per_hour = 4  # 15m = 4, 30m = 2, 60m = 1
startdate = dateutil.parser.parse('2017-06-27 12:00:00')  # goes from 26-06 to 05-07
enddate = dateutil.parser.parse('2017-07-02 12:00:00')
filename_data = '2017_dcount15m_harmonised.csv'
filename_pols = '2017_polygonlist.csv'
basepath = 'data/'


# create forecast model and plot, save plot under /img
def create_model_fbprophet(df_orig, polygon, hours_to_forecast, train_to_test_ratio):
    # prep cols for Prophet
    df_orig = df_orig[df_orig.polygon_id == int(polygon)]
    df_trimmed = df_orig.drop(['polygon_id'], axis=1)
    df_trimmed.columns = ['y', 'ds']
    # set date range
    df_trimmed.ds = pd.to_datetime(df_trimmed.ds, infer_datetime_format=True)
    df_trimmed = df_trimmed[(df_trimmed.ds >= startdate) & (df_trimmed.ds <= enddate)]
    # trim according to train/test split
    train_size = hours_to_forecast * slices_per_hour * train_to_test_ratio
    test_size = hours_to_forecast * slices_per_hour
    df_trimmed_train = df_trimmed[len(df_trimmed) - (train_size + 1):len(df_trimmed) - test_size]
    df_trimmed_test = df_trimmed[len(df_trimmed) - test_size:]
    # predict
    model = Prophet()
    model.fit(df_trimmed_train)
    print('model fitted.')

    return model, df_trimmed_test


def forecast_fbprophet(model, hours_to_forecast):
    future = model.make_future_dataframe(periods=hours_to_forecast, freq='H')
    forecast = model.predict(future)
    return forecast


def plot_fbprophet(model, polygon, forecast, hours_to_forecast, train_to_test_ratio, df_test):
    #plt.rcParams["figure.figsize"] = [3.0, 3.0]
    model.plot(forecast)
    # todo: add observed
    plt.savefig('img/forecast/' + 'fc_pol' + str(polygon) + '_' + str(hours_to_forecast) + '_ttr' + str(train_to_test_ratio) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    if hours_to_forecast == 24:
        model.plot_components(forecast)
        plt.savefig('img/forecast/' + 'decomp_pol' + str(polygon) + '_' + str(hours_to_forecast) + '_ttr' + str(train_to_test_ratio) + '.png', bbox_inches='tight')


if __name__ == '__main__':
    # setup
    train_to_test_ratio = 3
    hours_to_forecast = 24
    # load data
    df_orig = pd.read_csv(basepath + filename_data)

    # create forecasts
    for polygon in [24, 49]:    # Camping E and Orange Scene
        model, df_test = create_model_fbprophet(df_orig, polygon, hours_to_forecast, train_to_test_ratio)
        forecast = forecast_fbprophet(model, hours_to_forecast)
        plot_fbprophet(model, polygon, forecast, hours_to_forecast, train_to_test_ratio, df_test)
        print(hours_to_forecast)
