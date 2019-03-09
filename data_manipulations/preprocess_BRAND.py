import pandas as pd
import numpy as np
import math
from tqdm import tqdm 
import matplotlib.pyplot as plt 

data = pd.read_csv("beauty_data_info_train_competition.csv")

result = []

rows = np.asarray(data.iloc[:, :])
for i in tqdm(range(0,len(rows))):
    brand = rows[i][4]
    if not math.isnan(brand):
        result.append(rows[i])

for i in range(6):
    new_df = pd.DataFrame(result).to_csv("mixed_brands.csv", index=False, header=list(data.columns))

data = pd.read_csv("mixed_brands.csv")
brand = data['Brand'].value_counts()
blist = list(brand.index)
btimes = {blist[i]:i for i in range(len(blist))}

b_numerical = [btimes[e] for e in data['Brand']]
data['b_numerical'] = b_numerical 

data.to_csv("numerical_brands.csv", index=False, header=list(data.columns))
