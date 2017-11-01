import dateutil
import pandas as pd
import numpy as np
from fbprophet import Prophet
from matplotlib import pyplot

def getglobalslices():
    # load data
    df_global = pd.read_csv('data/2017_dcount15m_harmonised.csv', usecols=["timestamp", "devices"])
    df_global = df_global.groupby("timestamp")
    df_global = df_global.agg(np.sum)
    print(df_global.tail())
    df_global.to_csv("results/2017_global_1H.csv")

def getglobaldecomp():
    df_year = pd.read_csv('results/2017_global_1H.csv')
    df_year.columns = ['ds', 'y']
    df_year.plot()
    pyplot.show()

    model = Prophet()  # instantiate Prophet
    model.fit(df_year)  # fit the model with your dataframe
    future_data = model.make_future_dataframe(periods=1)
    forecast_data = model.predict(future_data)

    model.plot(forecast_data)
    pyplot.show()
    pyplot.clf()

    model.plot_components(forecast_data)
    pyplot.show()
    pyplot.clf()

getglobaldecomp()
