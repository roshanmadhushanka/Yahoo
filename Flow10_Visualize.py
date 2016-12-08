import matplotlib.pyplot as plt
import pandas as pd

# Load csv file
train = pd.read_csv('test.csv')

# Filter anomalies and normal data into different frames
anomaly_train = train.query('anomaly == 1')
normal_train = train.query('anomaly == 0')

# Define x and y axis [column names of the data frame]
x_axis = 'trend'
y_axis = 'seasonality3'

# Collect anomaly data
x_vals_anom = anomaly_train[x_axis]
y_vals_anom = anomaly_train[y_axis]

# Collect normal data
x_vals_norm = normal_train[x_axis]
y_vals_norm = normal_train[y_axis]


plt.scatter(x_vals_norm, y_vals_norm, c='w', alpha=0.5) # Plot normal data in white color
plt.scatter(x_vals_anom, y_vals_anom, c='r', alpha=0.5) # Plot anomaly data in red color

plt.legend(['normal', 'anomaly'])
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.show()