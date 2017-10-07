import matplotlib
from fbprophet import Prophet
import pandas as pd
from matplotlib.pyplot import show, savefig, errorbar, plot, clf, xlabel, ylabel, title, close
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
matplotlib.rcParams['axes.color_cycle'] = ['orange', 'lightblue', 'grey']

# setup forecast and cross-validation
fc_hours_to_predict = 2 # how far forecast looks into future
cv_horizon = '4 hour'   # how far cv looks into future -- options: hour, day, week, month

# setup data
nrows = 350
skiprows = range(1, 150)
datasets = ['raw15_SLICE15M_13.csv', 'raw15_SLICE15M_10.csv', 'raw15_SLICE15M_3.csv']
basepath = 'data/'
results = []

# create forecast model and plot, save plot under /img
def forecast(data):
    # dataset
    df = pd.read_csv(data, nrows=nrows, skiprows=skiprows)
    # prep for Prophet
    df.columns = ['ds', 'y']
    model = Prophet()
    # predict
    model.fit(df)
    print('model fitted.')
    future = model.make_future_dataframe(periods=fc_hours_to_predict * 4, freq='H')
    future.tail()
    print('df created.')
    forecast = model.predict(future)
    print('forecast made.')
    # plot
    model.plot(forecast)
    savefig('img/' + 'fc_' + file + '.png', bbox_inches='tight')
    close()
    clf()
    print('plot created.')
    return model

# calculate MAPE (timeseries evaluation metric)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# cross-validate prediction for model evaluation
def crossvalidate(model, filename):
    from fbprophet.diagnostics import cross_validation
    df_cv = cross_validation(model, horizon=cv_horizon)  # hour, day, week (singular)
    evaluate_cv(df_cv, filename)

def evaluate_cv(df_cv, filename):
    # report scores
    for hr_range in [24, 12, 8, 4, 2, 1]:
        # cut data
        df_cv = df_cv.iloc[0:hr_range*4]
        df_cv = df_cv[['ds', 'y', 'yhat']]

        # define metrics & create score
        MSE = mean_squared_error(df_cv.y, df_cv.yhat)
        R2 = r2_score(df_cv.y, df_cv.yhat)
        MAPE = mean_absolute_percentage_error(df_cv.y, df_cv.yhat)
        score = ('(%s) at %d hours - MSE: %.2f, R^2: %.2f, MAPE: %.2f pct' % (file, hr_range, MSE, R2, MAPE))
        # todo: uncertainty intervals/error bars?
        results.append(score)
        print(score)

        # plot
        df_cv.plot()
        xlabel('timestamp')
        ylabel('attendees')
        title(str(hr_range))
        savefig('img/' + 'cv_' + filename + '_' + str(hr_range) + 'h.png', bbox_inches='tight')
        close()
        clf()
    return results

# parallelise calculations
from joblib import Parallel, delayed
import multiprocessing

def processInput(file):
    data = basepath + file
    model = forecast(data, file)
    crossvalidate(model, file)
    return results

def parallelise():
    num_cores = multiprocessing.cpu_count()
    results.append(
        Parallel(n_jobs=num_cores)
        (delayed(processInput)(file)
         for file in datasets))

if __name__ == "__main__":
    parallelise()
    print(results)
