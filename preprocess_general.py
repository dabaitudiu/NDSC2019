import pandas as pd
import numpy as np
import math
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import pickle

filename = 'mobile_data_info_train_competition.csv'
label_index = 12
label_name = 'Camera'

# Pattern : 20
# Collar Type: 16
# Fashion Trend: 11
# Clothing Material 19
# Sleeves: 4
# Operating System: 7
# Features: 7
# Network Connections: 4
# Brand: 55
# Warranty Period: 14
# Storage Capacity: 8
# Color Family: 21
# Phone Model: 619
# Camera: 15

data = pd.read_csv(filename)

result = []

rows = np.asarray(data.iloc[:, :])
for i in tqdm(range(0,len(rows))):
    this_label = rows[i][label_index]
    if not math.isnan(this_label):
        result.append(rows[i])

print("Raw Rows: ",len(rows))

pd.DataFrame(result).to_csv("mobile_mixed_{}.csv".format(label_name), index=False, header=list(data.columns))

data = pd.read_csv("mobile_mixed_{}.csv".format(label_name))
this_label = data[label_name].value_counts()
blist = list(this_label.index)
btimes = {blist[i]:i for i in range(len(blist))}

b_numerical = [btimes[e] for e in data[label_name]]
data['numerical_{}'.format(label_name)] = b_numerical 

data.to_csv("mobile_numerical_{}.csv".format(label_name), index=False, header=list(data.columns))
pickle.dump(btimes,open('mobile_{}.pkl'.format(label_name),'wb'))

print("Left Rows: ",len(data))