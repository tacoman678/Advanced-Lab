import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

file_path = 'NMR Lab - Sheet1.csv'
df = pd.read_csv(file_path, usecols=['Frequency(MHz)', 'drift T/F', 't1', 't2', 't3', 't4', 't5', 'V1', 'V2', 'V3', 'V4', 'V5'])

frequencies = [21.68250, 21.60000, 21.68300, 21.60400, 21.68375, 21.61000, 21.61500, 21.62000, 21.62500, 21.63000, 21.63500, 21.64000, 21.64500, 21.65000, 21.70000, 21.65500, 21.66500, 21.66000]
y_columns = ['V1', 'V2', 'V3', 'V4', 'V5']
x_columns = ['t1', 't2', 't3', 't4', 't5']
decay_time = []

for f in frequencies:
    filtered_df = df[df['Frequency(MHz)'] == f]
    x = np.array([])
    y = np.array([])
    for v, t in zip(y_columns, x_columns):
        yn = filtered_df[v].to_numpy()
        xn = filtered_df[t].to_numpy()
        y = np.concatenate([y, yn])
        x = np.concatenate([x, xn])
    p0=[np.max(y)-np.min(y), np.log((np.max(y)-np.min(y))/np.min(np.abs(y-np.mean(y))))/(np.max(x)-np.min(x)),
    np.mean(y)]
    popt, pcov = curve_fit(func, x, y, p0=p0)
    decay_time.append([f, popt[2]])
decay_time_df = pd.DataFrame(decay_time, columns=['Frequency', '1/t'])
print(decay_time_df)

plt.scatter(decay_time_df['Frequency'], np.abs(1/decay_time_df['1/t']), marker='o')
# plt.plot(x, func(x, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' %tuple(popt))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Decay Time (microseconds)')
plt.legend()
plt.show()