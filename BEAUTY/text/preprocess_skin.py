import pandas as pd
import numpy as np
import math
from tqdm import tqdm 
import matplotlib.pyplot as plt 

data = pd.read_csv("beauty_data_info_train_competition.csv")

result = []

rows = np.asarray(data.iloc[:, :])
for i in tqdm(range(0,len(rows))):
    skin = rows[i][7]
    if not math.isnan(skin):
        result.append(rows[i])

pd.DataFrame(result).to_csv("mixed_skin.csv", index=False, header=list(data.columns))

data = pd.read_csv("mixed_skin.csv")
skin = data['Skin_type'].value_counts()
blist = list(skin.index)
btimes = {blist[i]:i for i in range(len(blist))}

b_numerical = [btimes[e] for e in data['Skin_type']]
data['skin_numerical'] = b_numerical 

data.to_csv("numerical_skins.csv", index=False, header=list(data.columns))
