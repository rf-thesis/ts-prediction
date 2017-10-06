import matplotlib
from fbprophet import Prophet
import pandas as pd
from matplotlib.pyplot import show
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
matplotlib.rcParams['axes.color_cycle'] = ['orange', 'lightblue', 'grey']

hours_to_predict = 24
cv_horizon = '4 hour'   # options: hour, day, week, month
data = 'data/raw15_SLICE15M_13.csv'
arr_data = ['data/raw15_SLICE15M_13.csv', 'data/raw15_SLICE15M_10.csv', 'data/raw15_SLICE15M_3.csv', 'data/raw15_SLICE15M_16.csv']
nrows = 100

def predict(data):
    # dataset
    df = pd.read_csv(data, nrows=nrows)
    # prep for Prophet
    df.columns = ['ds', 'y']
    model = Prophet()
    # predict
    model.fit(df)
    print('model fitted.')
    future = model.make_future_dataframe(periods=hours_to_predict * 4, freq='H')
    future.tail()
    print('df created.')
    forecast = model.predict(future)
    print('forecast made.')
    # plot
    model.plot(forecast)
    # model.plot_components(forecast)
    print('plot created.')
    return model


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def crossvalidate():
    from fbprophet.diagnostics import cross_validation
    df_cv = cross_validation(model, horizon=cv_horizon)  # hour, day, week (singular)
    df_cv = df_cv[['y', 'yhat']]
    y_true, y_pred = df_cv['y'], df_cv['yhat']
    MSE = mean_squared_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    MAPE = mean_absolute_percentage_error(y_true, y_pred)
    print('MSE: %.2f, R^2: %.2f, MAPE: %.2f pct' % (MSE, R2, MAPE))
    df_cv.plot()

for dataset in arr_data:
    model = predict(dataset)
    crossvalidate(model)
    show()  # show all plots
