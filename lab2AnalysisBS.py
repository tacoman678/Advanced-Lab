import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

proton_moment = 1.410e-26
h = 6.63e-34

file_path = 'NMR Lab - Sheet1.csv'
df = pd.read_csv(file_path, usecols=['Frequency(MHz)','drift time', 'drift T/F', 't1', 't2'])

filtered_df = df[(df['drift T/F'] == 1) & (df['Frequency(MHz)'] == 21.658)]
filtered_df.loc[:, 'cum time'] = filtered_df['drift time'].cumsum()
filtered_df.loc[:, 'T'] = (filtered_df['t2']* 10**-6 - filtered_df['t1']* 10**-6)
filtered_df.loc[:, 'f0_un'] = (20e-6)/(filtered_df['T'])
filtered_df.loc[:, 'B'] = (h/(2*proton_moment)) * (filtered_df['Frequency(MHz)'] * 10**6 -(1/filtered_df['T']))
filtered_df.loc[:, 'B_un'] = ((20/(filtered_df['Frequency(MHz)'] * 10**6))**2 + (filtered_df['f0_un']/(1/filtered_df['T']))**2)** .5
print(filtered_df)

plt.errorbar(filtered_df['cum time'], filtered_df['B'], yerr=filtered_df['B_un'], marker='o', color='blue', ls='none')
fit = np.polyfit(filtered_df['cum time'], filtered_df['B'], 1)
plt.plot(filtered_df['cum time'], fit[0] * filtered_df['cum time'] + fit[1], color='blue', linestyle='--', label='Strength of Magnetic Field vs Time')
plt.xlabel('Cumulative Drift time (s)')
plt.ylabel('Strength of Magnetic Field (T)')
plt.legend()

plt.show()
print("Slope of the magnetic field strength vs time: ", fit[0])
print("actual field stability: ", (5e-7/(15*60)))