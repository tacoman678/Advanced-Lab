import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = 'NMR Lab - Sheet1.csv'
df = pd.read_csv(file_path, usecols=['Frequency(MHz)', 'drift T/F', 't1', 't5'])

filtered_df = df[(df['drift T/F'] == 0)]
filtered_df.loc[:, 'Fd'] = filtered_df['Frequency(MHz)']
filtered_df.loc[:, 'T'] = ((filtered_df['t5']* 10**-6) - (filtered_df['t1']* 10**-6))/5
filtered_df.loc[:, 'Fo'] = (1/filtered_df['T'])/10**6
filtered_df.loc[:, 'Fo_un'] = (1e-6)/(filtered_df['T']**2)*10e-6
filtered_df.loc[:, 'Fd_un'] = 20/10**6
print(filtered_df)

plt.errorbar(filtered_df['Fo'], filtered_df['Fd'], xerr=filtered_df['Fo_un'], yerr=filtered_df['Fd_un'], marker='o', ls='none')
fit = np.polyfit(filtered_df['Fo'], filtered_df['Fd'], 1)
plt.plot(filtered_df['Fo'], fit[0] * filtered_df['Fo'] + fit[1], color='blue', linestyle='--')
plt.xlabel('Frequency Observed (MHz)')
plt.ylabel('Drive Frequency (MHZ)')
plt.legend()

plt.show()
print("Frequency Observed vs Drive Frequency Slope: ", fit[0])
print("Resonance Frequency: ", fit[1])
