import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.odr

def f(B,t):
    return B[0]*np.exp(-B[1]*t)+B[2]

#datasets 2-4 are 5 microsecond x-scale, datasets 5-8 are 20 microsecond x-scale
datasets = ["Data_Set_2.data","Data_Set_3.data","Data_Set_4.data","Data_Set_5.data","Data_Set_6.data","Data_Set_7.data","Data_Set_8.data"]
l5 = pd.DataFrame()
g5 = pd.DataFrame()
for i in range(0,6):
    if(i<3):
        l5 = pd.concat([l5,pd.read_csv(datasets[i], sep= " ", header=None, names=["pulse", "time"])])
    else:
        g5 = pd.concat([g5,pd.read_csv(datasets[i], sep= " ", header=None, names=["pulse", "time"])])

# dataset 5 microsecond x-sclae fit
filtered_df = l5[(l5["pulse"] < 5000)]
nums, time_bins = np.histogram(filtered_df["pulse"], bins=10)
avg_time_bins = []
for i in range(0,len(time_bins)-1):
    avg_time_bins.append((time_bins[i]+time_bins[i+1])/2)
odr_data = scipy.odr.RealData(np.array(avg_time_bins), nums)
myodr = scipy.odr.ODR(odr_data, scipy.odr.Model(f), beta0 = [1000.,1/2500.,100.])
output = myodr.run()
output.pprint()

# dataset 5 microsecond x-scale with fit
plt.hist(filtered_df["pulse"], bins=10)
plt.plot(time_bins,f(output.beta,time_bins))
print("average muon lifetime in microseconds (5 microsecond x-scale):", 1/output.beta[1])
print("Constant:", output.beta[2])
print("relative speed of muon", math.sqrt(1-(2197**2/((1/output.beta[1])**2)))*3e8)
print("speed relative to height(100km)", math.sqrt(1/((2.197e-6/100e3)*3e8 + 1)))
print("speed relative to height(10km)", math.sqrt(1/((2.197e-6/10e3)*3e8 + 1)))
# plt.show()

# dataset 20 microsecond x-scale fit
filtered_df1 = g5[(g5["pulse"] < 20000)]
nums, time_bins = np.histogram(filtered_df1["pulse"], bins=40)
avg_time_bins = []
for i in range(0,len(time_bins)-1):
    avg_time_bins.append((time_bins[i]+time_bins[i+1])/2)
odr_data = scipy.odr.RealData(np.array(avg_time_bins), nums)
myodr = scipy.odr.ODR(odr_data, scipy.odr.Model(f), beta0 = [1000.,1/2500.,100.])
output = myodr.run()
output.pprint()

# dataset 20 microsecond x-scale with fit
plt.hist(filtered_df1["pulse"], bins=40)
plt.plot(time_bins,f(output.beta,time_bins))
print("average muon lifetime in microseconds (20 microsecond x-scale):", 1/output.beta[1])
print("Constant:", output.beta[2])
# plt.show()

#both datasets 5 microsecond x-scale fit
all5 = pd.concat([filtered_df, g5[(g5["pulse"] < 5000)]])
nums, time_bins = np.histogram(all5["pulse"], bins=10)
avg_time_bins = []
for i in range(0,len(time_bins)-1):
    avg_time_bins.append((time_bins[i]+time_bins[i+1])/2)
odr_data = scipy.odr.RealData(np.array(avg_time_bins), nums)
myodr = scipy.odr.ODR(odr_data, scipy.odr.Model(f), beta0 = [1000.,1/2500.,100.])
output = myodr.run()
output.pprint()

# dataset 20 microsecond x-scale with fit
plt.hist(all5["pulse"], bins=10)
plt.plot(time_bins,f(output.beta,time_bins))
print("average muon lifetime in microseconds (all 5 microsecond x-scale):", 1/output.beta[1])
# plt.show()
