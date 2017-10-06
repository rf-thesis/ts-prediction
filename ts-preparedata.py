import pandas as pd

# define data
polygon = 16
nrows = None
skiprows = None
slice_col = 'SLICE15M'   # SLICE1H or 60m-slice
uid_col = 'USER_ID'
HIT_col = 'HIT'
ts_col = 'REAL_TIMESTAMP'
data    = 'data/raw15.csv'
output  = 'data/raw15_' + slice_col + '_' + str(polygon) + '.csv'

# read data
df = pd.read_csv(data, usecols=[slice_col, uid_col, HIT_col, ts_col], nrows=nrows, skiprows=skiprows)    # usecols for memory optim.

# filter
if polygon: df = df[(df.HIT == polygon)]
#df = df[(df[slice_col] >= 7000000)]

# create DF
# find first timestamp for each slice
df_ts = df.groupby([slice_col])[ts_col].min()
df_ts = df_ts.to_frame()
print(df_ts.head())
# count unique users per slice
df_count = df.groupby(slice_col)[uid_col].nunique()
df_count = df_count.to_frame()  # convert the above created Series to a DF
print(df_count.head())
# join both DFs
df_join = pd.concat([df_ts, df_count], axis=1)
print(df_join.head())

# name cols
df_join.reset_index(level=0, inplace=True)
df_join.columns = ['SLICE', 'TIMESTAMP', 'COUNT']
df_join.drop('SLICE', axis=1, inplace=True)

# remove outliers by cutting of upper/lower quantiles
# remove quantiles
def remOutliers(quantile, df):
    print('before outlier removal', len(df))
    q_up = df.COUNT.quantile(quantile)
    q_down = df.COUNT.quantile(1-quantile)
    df = df[df.COUNT < q_up]
    df = df[df.COUNT > q_down]
    print('after outlier removal', len(df), q_up, q_down)
    return df

#df_join = remOutliers(0.95, df_join)

# write csv
print(output)
df_join.to_csv(output, index=False)
