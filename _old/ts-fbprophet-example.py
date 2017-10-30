from fbprophet import Prophet
from matplotlib.pyplot import show
import pandas as pd

df = pd.read_csv('data/outdoor-temperature-hourly.csv')
df = df[df.temperature != 'DIFF']

# prep DF for Prophet (timestamp (ds) and target (y))
df['ds'] = df['time']
df['y'] = df['temperature']

# drop unnecessary stuff
df = df.drop(['name', 'time', 'temperature', 'seriesA', 'seriesB'], axis=1)

# instantiate & fit the model
model = Prophet()
model.fit(df)

print('model fitted.')

# forecast, given a specific future timespan (periods) -- freq 'H'
future = model.make_future_dataframe(periods=24*1, freq='H')
future.tail()
print('df created.')
forecast = model.predict(future)
print('forecast made.')

# plot results
model.plot(forecast)
# plot components
model.plot_components(forecast)
# show plot
show()

print('plot created.')