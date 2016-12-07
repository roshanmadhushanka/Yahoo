from featureeng.presenting import Chart
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

test = pd.read_csv('test.csv')
scaler = MinMaxScaler()
scaled = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)
# value, trend, seasonality1, noise
Chart.presentData(scaled, ['trend','value'])
