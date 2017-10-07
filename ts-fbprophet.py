
import dateutil
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
plt.rcParams['axes.color_cycle'] = ['orange', 'lightblue', 'grey']

# setup forecast
fc_hours_to_predict = 2 # how far forecast looks into future
# setup cv
cv_horizon_amount = 24       # how far cv looks into future
cv_horizon_unit = 'hour'    # units: sec, minute, hour, day, week, month
hr_range_arr = [24, 12, 8, 4, 2, 1] # state model accuracy for these hours for RMSE, R^2, MAPE

# setup data
slices_per_hour = 4         # 15m = 4, 30m = 2, 60m = 1
startdate = dateutil.parser.parse('2015-06-28 00:00:00') # goes from 26-06 to 05-07
enddate = dateutil.parser.parse('2015-07-03 00:00:00')
nrows = None
skiprows = None
datasets = ['raw15_SLICE15M_13.csv', 'raw15_SLICE15M_10.csv', 'raw15_SLICE15M_3.csv']
basepath = 'data/'
results = []


# create forecast model and plot, save plot under /img
def forecast(data, filename):
    # dataset
    df = pd.read_csv(data, nrows=nrows, skiprows=skiprows)
    # prep cols for Prophet
    df.columns = ['ds', 'y']
    # set date range
    df.ds = pd.to_datetime(df.ds, infer_datetime_format=True)
    df = df[(df.ds >= startdate) & (df.ds <= enddate)]
    # predict
    model = Prophet()
    model.fit(df)
    print('model fitted.')
    future = model.make_future_dataframe(periods=fc_hours_to_predict * 4, freq='H')
    future.tail()
    print('df created.')
    forecast = model.predict(future)
    print('forecast made.')
    # plot
    model.plot(forecast)
    plt.savefig('img/' + 'fc_' + filename + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    print('plot created.')
    return model

# calculate MAPE (timeseries evaluation metric)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# cross-validate prediction for model evaluation
def crossvalidate(model, filename):
    from fbprophet.diagnostics import cross_validation
    cv_horizon = str(cv_horizon_amount) + ' ' + cv_horizon_unit
    df_cv = cross_validation(model, horizon=cv_horizon)  # hour, day, week (singular)
    evaluate_cv(df_cv, filename)

def evaluate_cv(df_cv, filename):
    # report scores
    for hr_range in hr_range_arr:
        # cut data
        df_cv = df_cv.iloc[0:hr_range*slices_per_hour]
        df_cv = df_cv[['ds', 'y', 'yhat']]

        # define metrics & create score
        MSE = mean_squared_error(df_cv.y, df_cv.yhat)
        R2 = r2_score(df_cv.y, df_cv.yhat)
        MAPE = mean_absolute_percentage_error(df_cv.y, df_cv.yhat)
        score = ('(%s) at %d hours - MSE: %.2f, R^2: %.2f, MAPE: %.2f pct' % (filename, hr_range, MSE, R2, MAPE))
        # todo: uncertainty intervals/error bars?
        results.append(score)
        print(score)

        # plot
        df_plot = df_cv.set_index('ds')   # set timestamp as index
        df_plot = df_plot.cumsum()
        plt.figure
        df_plot.plot()
        plt.xlabel('timestamp')
        plt.ylabel('attendees')
        plt.title('hours considered: ' + str(hr_range))
        plt.savefig('img/' + 'cv_' + filename + '_' + str(hr_range) + 'h.png', bbox_inches='tight')
        plt.close()
        plt.clf()
    return results

# parallelise calculations
from joblib import Parallel, delayed
import multiprocessing

def processInput(filename):
    data = basepath + filename
    model = forecast(data, filename)
    crossvalidate(model, filename)
    return results

def parallelise():
    num_cores = multiprocessing.cpu_count()
    results.append(
        Parallel(n_jobs=num_cores)
        (delayed(processInput)(filename)
         for filename in datasets))

if __name__ == "__main__":
    parallelise()
    print(results)
