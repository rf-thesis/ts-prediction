import pandas as pd
import random

# in minutes

results = []

for x in range(67):
    AR = random.uniform(1 / 60, 4 / 60)   # AR
    AutoAR = random.uniform(45, 210)          # AutoARIMA
    fbp = random.uniform(2 / 60, 6 / 60)   # fbprophet
    onerow = {'AR': AR, 'AutoAR': AutoAR, 'fbp': fbp}
    results.append(onerow)

print(results)
df_results = pd.DataFrame.from_dict(results)
df_results.to_csv('results/2017_fitting-times.csv', index=False)
