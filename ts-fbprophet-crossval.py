import dateutil
from fbprophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Manager
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# setup forecast
fc_hours_to_predict = 24  # how far forecast looks into future
# setup cv
run_cv = True  # run CV yes/no
cv_horizon_amount = 24  # how far cv looks into future
cv_horizon_unit = 'hour'  # units: sec, minute, hour, day, week, month
hr_range_arr = [32, 24, 16, 12, 8, 4, 2, 1]  # state model accuracy for these hours for RMSE, R^2, MAPE

# setup data
plotforecasts, plotcrossvals, plotcvuncertainty = False, False, False
slices_per_hour = 4  # 15m = 4, 30m = 2, 60m = 1
startdate = dateutil.parser.parse('2017-06-27 12:00:00')  # goes from 26-06 to 05-07
enddate = dateutil.parser.parse('2017-07-02 12:00:00')
filename_data = '2017_dcount15m_harmonised.csv'
filename_pols = '2017_polygonlist.csv'
basepath = 'data/'


# create forecast model and plot, save plot under /img
def create_model_fbprophet(df_orig, polygon):
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

    return model


def plot_fbprophet(model, polygon):
    future = model.make_future_dataframe(periods=fc_hours_to_predict * 4, freq='H')
    future.tail()
    print('df created.')
    forecast = model.predict(future)
    print('forecast made.')

    # plot
    if plotforecasts:
        model.plot(forecast)
        plt.savefig('img/forecast/' + 'fc_' + filename_data + '_pol' + str(polygon) + '.png', bbox_inches='tight')
        plt.close()
        plt.clf()
        print('plot created.')

    return forecast


# cross-validate prediction for model evaluation
def crossval_fbprophet(model, polygon):
    from fbprophet.diagnostics import cross_validation
    cv_horizon = str(cv_horizon_amount) + ' ' + cv_horizon_unit
    df_cv = cross_validation(model, horizon=cv_horizon)  # hour, day, week (singular)
    return df_cv


# calculate MAPE (timeseries evaluation metric)
def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# calculate SMAPE (timeseries evaluation metric)
def calc_SMAPE(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 200 * np.mean(diff)


def evaluate_cv_per_hour(df_cv, polygon):
    list_cvscore_per_hr = []

    # report scores for each hour range
    for hr_range in hr_range_arr:
        # cut data for each hour range
        df_cv = df_cv.iloc[0:hr_range * slices_per_hour]
        # define metrics & create score
        RMSE = np.sqrt(mean_squared_error(df_cv.y, df_cv.yhat))
        R2 = r2_score(df_cv.y, df_cv.yhat)
        SMAPE = calc_SMAPE(df_cv.y, df_cv.yhat)
        list_cvscore_per_hr.append({'POLYGON': polygon, 'HOUR': hr_range, 'RSQUARED': R2, 'RMSE': RMSE, 'SMAPE': SMAPE})

    return list_cvscore_per_hr

# plot the crossvalidation results against the observed data (unused)
def plot_crossval(df_cv, polygon):
    # set timestamp as index and convert to datetime
    df_plot = df_cv.set_index('ds')
    df_plot.index = pd.to_datetime(df_plot.index, infer_datetime_format=True)
    # plot functions
    if plotcrossvals:
        plt.figure
        df_plot.plot()
        plt.fill_between(df_plot.index, df_plot.yhat_lower, df_plot.yhat_upper, facecolor='b', edgecolor='#1B2ACC',
                         antialiased=True, alpha=.1)
        plt.xlabel('timestamp')
        plt.ylabel('attendees')
        plt.title('Cross-Validation of Polygon ' + str(polygon))
        plt.savefig('img/' + 'cv_' + filename_data + '_pol' + str(polygon) + '.png', bbox_inches='tight')
        plt.close()
        plt.clf()
    if plotcvuncertainty:
        fig = plt.figure(0)
        df_plot.yhat.plot()
        df_plot.y.plot()
        plt.fill_between(df_plot.index, df_plot.yhat_lower, df_plot.yhat_upper, interpolate=False, facecolor='b',
                         edgecolor='#1B2ACC', antialiased=True, alpha=.1)
        # plt.plot(df_plot.index, df_plot.y)
        # plt.plot(df_plot.index, df_plot.yhat)
        # plt.errorbar(df_plot.index, df_plot.yhat, yerr=[y-df_plot.yhat_upper, y-df_plot.yhat_lower], uplims=True, lolims=True)
        plt.legend(['y', 'y_hat', 'uncertainty'])
        plt.xlabel('timestamp')
        plt.ylabel('attendees')
        plt.title('Cross-Validation of Polygon ' + str(polygon))
        plt.savefig('img/' + 'cv_uc_' + filename_data + '_pol' + str(polygon) + '.png', bbox_inches='tight')
        plt.close()
        plt.clf()

# parallelise calculations
from joblib import Parallel, delayed
import multiprocessing


def predict_one_polygon(df_orig, polygon):
    model = create_model_fbprophet(df_orig, polygon)
    df_cv = crossval_fbprophet(model, polygon)
    cvscore_per_hr = evaluate_cv_per_hour(df_cv, polygon)
    #df_cv.to_csv('so-data' + str(polygon) + '.csv')
    return cvscore_per_hr


def parallelise(df_orig, list_polygons, allresults):
    num_cores = multiprocessing.cpu_count()
    # update the allresults list
    # use [0] to flatten array
    allresults.extend(
        Parallel(n_jobs=num_cores)(delayed(predict_one_polygon)(df_orig, polygon) for polygon in list_polygons))


def create_final_df(allresults):
    # create pandas df from dict
    df_endresult = pd.DataFrame()
    for i in range(0, len(allresults)):
        one_row = pd.DataFrame.from_dict(allresults[i])
        df_endresult = df_endresult.append(one_row)
    df_endresult = df_endresult.set_index('POLYGON')
    # join with polygon names list
    df_pinfo = pd.read_csv('data/2017_polygoninfo_wout1.csv', usecols=['ogr_fid', 'name'])
    df_pinfo = df_pinfo.set_index('ogr_fid')
    df_pinfo.index.rename('POLYGON', inplace=True)
    df_joined = df_endresult.join(df_pinfo, how='outer')
    return df_joined


if __name__ == "__main__":
    # TODO: CHANGE TO 'None' AFTER DEBUGGING
    nrows = None

    # load data
    df_orig = pd.read_csv(basepath + filename_data)
    df_polygons = pd.read_csv(basepath + filename_pols, nrows=nrows)
    list_polygons = df_polygons.values.flatten()
    print(list_polygons)
    # parallelise processing
    allresults = Manager().list([])  # enable memory sharing between processes
    parallelise(df_orig, list_polygons, allresults)
    # create a df of all dicts, join with polygon names
    df_final = create_final_df(allresults)
    df_final.to_csv('results/2017_fbprophet-cvresults.csv')
    print(df_final.head(n=10))
