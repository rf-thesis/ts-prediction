from matplotlib.pyplot import show

df = pd.read_csv('data/outdoor-temperature-hourly.csv')
df = df[df.temperature != 'DIFF']
df['ds'] = df['time']
df['y'] = df['temperature']
df = df.drop(['name', 'time', 'temperature', 'seriesA', 'seriesB'], axis=1)

model = Prophet()
model.fit(df)

print('model fitted.')

future = model.make_future_dataframe(periods=24*1, freq='H')
future.tail()

print('df created.')

forecast = model.predict(future)

print('forecast made.')

model.plot(forecast)
show()

print('plot created.')