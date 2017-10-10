
import dateutil
import numpy
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
plt.rcParams['axes.color_cycle'] = ['orange', 'lightblue', 'grey']

# setup forecast
fc_hours_to_predict = 2 # how far forecast looks into future
# setup cv
run_cv = True               # run CV yes/no
cv_horizon_amount = 24       # how far cv looks into future
cv_horizon_unit = 'hour'    # units: sec, minute, hour, day, week, month
hr_range_arr = [24, 12, 8, 4, 2, 1] # state model accuracy for these hours for RMSE, R^2, MAPE

# setup data
plotforecasts, plotcrossvals, plotcvuncertainty = True, False, False
slices_per_hour = 4                                         # 15m = 4, 30m = 2, 60m = 1
startdate = dateutil.parser.parse('2017-06-28 12:00:00')    # goes from 26-06 to 05-07
enddate =   dateutil.parser.parse('2017-07-03 12:00:00')
filename_data = '2017_devicecount15m.csv'
filename_pols = '2017_devicecount15m-polygons.csv'
basepath = 'data/'
results = []

# create forecast model and plot, save plot under /img
def forecast(df_orig, polygon):
    # prep cols for Prophet
    df_orig = df_orig[df_orig.polygon_id == int(polygon)]
    df_trimmed = df_orig.drop(['polygon_id'], axis=1)
    df_trimmed.columns = ['y', 'ds']
    # set date range
    df_trimmed.ds = pd.to_datetime(df_trimmed.ds, infer_datetime_format=True)
    df_trimmed = df_trimmed[(df_trimmed.ds >= startdate) & (df_trimmed.ds <= enddate)]
    # predict
    model = Prophet()
    model.fit(df_trimmed)
    print('model fitted.')
    future = model.make_future_dataframe(periods=fc_hours_to_predict * 4, freq='H')
    future.tail()
    print('df created.')
    forecast = model.predict(future)
    print('forecast made.')

    # plot
    if plotforecasts:
        model.plot(forecast)
        plt.savefig('img/' + 'fc_' + filename_data + '_pol' + str(polygon) + '.png', bbox_inches='tight')
        plt.close()
        plt.clf()
        print('plot created.')

    return model

# calculate MAPE (timeseries evaluation metric)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# cross-validate prediction for model evaluation
def crossvalidate(model, polygon):
    from fbprophet.diagnostics import cross_validation
    cv_horizon = str(cv_horizon_amount) + ' ' + cv_horizon_unit
    try:
        df_cv = cross_validation(model, horizon=cv_horizon)  # hour, day, week (singular)
        evaluate_cv(df_cv, polygon)
    except ValueError:
        results.append({'POLYGON': None, 'HOUR': None, 'RSQUARED': None, 'RMSE': None, 'MAPE': None})
        print('Not enough data for specified horizon. Decrease horizon or change period/initial.')

def evaluate_cv(df_cv, polygon):
    # plot
    # set timestamp as index and convert to datetime
    df_plot = df_cv.set_index('ds')
    df_plot.index = pd.to_datetime(df_plot.index, infer_datetime_format=True)
    # plot functions
    if plotcrossvals:
        plt.figure
        df_plot.plot()
        plt.xlabel('timestamp')
        plt.ylabel('attendees')
        plt.title('Cross-Validation of Polygon ' + str(polygon))
        plt.savefig('img/' + 'cv_' + filename_data + '_pol' + str(polygon) + '.png', bbox_inches='tight')
        plt.close()
        plt.clf()
    if plotcvuncertainty:
        fig = plt.figure(0)
        plt.plot(df_plot.index, df_plot.y)
        plt.plot(df_plot.index, df_plot.yhat)
        #plt.errorbar(df_plot.index, df_plot.yhat, yerr=[y-df_plot.yhat_upper, y-df_plot.yhat_lower], uplims=True, lolims=True)
        plt.fill_between(df_plot.index, df_plot.yhat_upper, df_plot.yhat_lower, facecolor='b', alpha=.1)
        plt.legend(['y', 'y_hat', 'uncertainty'])
        plt.xlabel('timestamp')
        plt.ylabel('attendees')
        plt.title('Cross-Validation of Polygon ' + str(polygon))
        plt.savefig('img/' + 'cv_uc_' + filename_data + '_pol' + str(polygon) + '.png', bbox_inches='tight')
        plt.show()
        plt.close()
        plt.clf()

    # report scores for each hour range
    for hr_range in hr_range_arr:
        # cut data for each hour range
        df_cv = df_cv.iloc[0:hr_range*slices_per_hour]
        # define metrics & create score
        RMSE = mean_squared_error(df_cv.y, df_cv.yhat)
        R2 = r2_score(df_cv.y, df_cv.yhat)
        MAPE = mean_absolute_percentage_error(df_cv.y, df_cv.yhat)
        results.append({'POLYGON': polygon, 'HOUR': hr_range, 'RSQUARED': R2, 'RMSE': RMSE, 'MAPE': MAPE})

    return results

# parallelise calculations
from joblib import Parallel, delayed
import multiprocessing

def runOneProcess(df_orig, polygon):
    model = forecast(df_orig, polygon)
    if run_cv:
        if model: crossvalidate(model, polygon)
        return results

def parallelise(df_orig, df_polygons):
    num_cores = multiprocessing.cpu_count()
    results.append(
        Parallel(n_jobs=num_cores)
        (delayed(runOneProcess)(df_orig, polygon)
         for polygon in df_polygons))

if __name__ == "__main__":
    # load data
    df_orig = pd.read_csv(basepath + filename_data)
    df_polygons = pd.read_csv(basepath + filename_pols, nrows=1)   # TODO:
    list_polygons = df_polygons.values.flatten()
    # parallelise processing
    #runOneProcess(df_orig, 13)
    parallelise(df_orig, list_polygons) # TODO: change for debug
    # output data
    print(results)
    df_results = pd.DataFrame(results[0][0])    # this list gets nested 2 times (why????) , therefore [0][0]
    df_results = df_results.set_index('POLYGON')
    df_results.to_csv('results.csv')
    print(df_results.head())
