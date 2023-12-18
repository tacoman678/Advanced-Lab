import pandas as pd
import datetime as dt
import numpy as np

#datasets 2-4 are 5 microsecond x-scale, datasets 5-8 are 20 microsecond x-scale
datasets = ["Data_Set_2.data","Data_Set_3.data","Data_Set_4.data","Data_Set_5.data","Data_Set_6.data","Data_Set_7.data","Data_Set_8.data"]
l5 = pd.DataFrame()
g5 = pd.DataFrame()
for i in range(0,6):
    if(i<3):
        l5 = pd.concat([l5,pd.read_csv(datasets[i], sep= " ", header=None, names=["pulse", "time"])])
    else:
        g5 = pd.concat([g5,pd.read_csv(datasets[i], sep= " ", header=None, names=["pulse", "time"])])

l5_decays = l5[(l5["pulse"] < 5000)]
l5_head = l5.iloc[0]["time"]
l5_tail = l5.iloc[-1]["time"]
print("muons decayed per second (5 microsecond x-scale):", len(l5_decays) / (l5_tail - l5_head))

g5_decays = g5[(g5["pulse"] < 20000)]
g5_head = g5.iloc[0]["time"]
g5_tail = g5.iloc[-1]["time"]
print("muons decayed per second (20 microsecond x-scale):", len(g5_decays) / (g5_tail - g5_head))

print("background events (5 microsecond x-scale):", (len(l5[(l5["pulse"] <= 20000)])))
print("background events (20 microsecond x-scale):", (len(g5[(g5["pulse"] <= 20000)])))

print("background events per time bin", (len(l5[(l5["pulse"] <= 20000)]))*.5e-6)
print("background events per time bin", (len(g5[(g5["pulse"] <= 20000)]))*.5e-6)

lamb_5 = (len(l5_decays)*.5)*(1/2322.2805794982933) + (len(l5_decays)*.5)*(1/2322.2805794982933 + (1e-6/26))/(len(l5_decays))
print(lamb_5)

lamb_20 = (len(g5_decays)*.5)*(1/2064.790988606895) + (len(g5_decays)*.5)*(1/2064.790988606895 + (1e-6/26))/(len(g5_decays))
print(lamb_20)