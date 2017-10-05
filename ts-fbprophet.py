import matplotlib
import sklearn
from fbprophet import Prophet
import pandas as pd
import numpy as np
from matplotlib.pyplot import show
matplotlib.rcParams['axes.color_cycle'] = ['orange', 'lightblue', 'grey']

periods_to_predict = 10

# dataset
df = pd.read_csv('data/raw15_SLICE15M_None.csv')

# prep for Prophet
df.columns = ['ds', 'y']

# predict
model = Prophet()
model.fit(df)

print('model fitted.')

future = model.make_future_dataframe(periods=periods_to_predict, freq='H')
future.tail()

print('df created.')

forecast = model.predict(future)

print('forecast made.')

model.plot(forecast)
model.plot_components(forecast)

print('plot created.')

from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(model, horizon = '1 hour') # hour, day, week (singular)
df_cv = df_cv[['y', 'yhat']]

y_true, y_pred = df_cv['y'], df_cv['yhat']
mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
print(mse)

df_cv.plot()
show()
