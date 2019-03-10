import pandas as pd
import numpy as np
import math
from tqdm import tqdm 
import matplotlib.pyplot as plt 

data = pd.read_csv("beauty_data_info_train_competition.csv")

result = []

rows = np.asarray(data.iloc[:, :])
for i in tqdm(range(0,len(rows))):
    texture = rows[i][6]
    if not math.isnan(texture):
        result.append(rows[i])

pd.DataFrame(result).to_csv("mixed_texture.csv", index=False, header=list(data.columns))

data = pd.read_csv("mixed_texture.csv")
texture = data['Product_texture'].value_counts()
blist = list(texture.index)
btimes = {blist[i]:i for i in range(len(blist))}

b_numerical = [btimes[e] for e in data['Product_texture']]
data['texture_numerical'] = b_numerical 

data.to_csv("numerical_textures.csv", index=False, header=list(data.columns))
