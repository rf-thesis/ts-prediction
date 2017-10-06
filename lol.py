import pandas as pd

df = pd.read_csv('data/raw15.csv', usecols=['HIT'], nrows=300000)
df = df.groupby(['HIT'])[['HIT']].size()
print(df)
df.to_csv('data/mostHITs_2015.csv')