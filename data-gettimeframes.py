import dateutil
import pandas as pd
import matplotlib.pyplot as plt

startdate = dateutil.parser.parse('2017-06-26 12:00:00')    # goes from 26-06 to 05-07
enddate =   dateutil.parser.parse('2017-06-30 12:00:00')

filename_data = '2017_devicecount15m.csv'
filename_pols = '2017_devicecount15m-polygons.csv'
basepath = 'data/'

#polygon_id,devices,timestamp

# load data
df_orig = pd.read_csv(basepath + filename_data)
df_polygons = pd.read_csv(basepath + filename_pols, nrows=None)
list_polygons = df_polygons.values.flatten()

# remove outliers by cutting of upper/lower quantiles
# remove quantiles
# def remOutliers(quantile, df):
#     print('before outlier removal: count %.2f' % len(df))
#     q_up = df.devices.quantile(quantile)
#     q_down = df.devices.quantile(1-quantile)
#     df = df[df.devices < q_up]
#     df = df[df.devices > q_down]
#     print('after outlier removal: count %.2f q_up %.2f q_down %.2f' % (len(df), q_up, q_down))
#     return df

results = []
for polygon in list_polygons:
    df_this = df_orig[df_orig.polygon_id == int(polygon)]
    ts_min = df_this.timestamp.min()
    ts_max = df_this.timestamp.max()
    results.append({'polygon': polygon, 'ts_min': ts_min, 'ts_max': ts_max})

df_minmax = pd.DataFrame(results)
df_minmax.to_csv('df-minmax.csv', index=False)
