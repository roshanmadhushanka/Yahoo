import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('test.csv')

anomaly_train = train.query('anomaly == 1')
normal_train = train.query('anomaly == 0')

x_axis = 'trend'
y_axis = 'value'

x_vals_anom = anomaly_train[x_axis]
y_vals_anom = anomaly_train[y_axis]

x_vals_norm = normal_train[x_axis]
y_vals_norm = normal_train[y_axis]

plt.scatter(x_vals_norm, y_vals_norm, c='w', alpha=0.5)
plt.scatter(x_vals_anom, y_vals_anom, c='r', alpha=0.5)

plt.legend(['normal', 'anomaly'])
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.show()