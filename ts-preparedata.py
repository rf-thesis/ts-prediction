import pandas as pd

# define data
data = 'slices_2017.csv'
slice_col = '60m-slice' # SLICE1H or 60m-slice or slice-60m
uid_col = 'device_id'
output = 'slices_2017_1H.csv'

# read data
df = pd.read_csv(data, usecols=[slice_col, uid_col])    # usecols for memory optim.
# group by timeslice, ignore duplicate devices
df = df.groupby(slice_col)[uid_col].nunique()
df = df.to_frame()  # convert the above created Series to a DF

# remove outliers by cutting of upper/lower quantiles
print 'before outlier removal', len(df)
# name cols
df.index.name = 'SLICE'
df.columns = ['COUNT']
# remove quantiles
q_up = df.COUNT.quantile(0.95)
q_down = df.COUNT.quantile(0.05)
df = df[df.COUNT < q_up]
df = df[df.COUNT > q_down]
df.to_csv(output)
print 'after outlier removal', len(df), q_up, q_down
