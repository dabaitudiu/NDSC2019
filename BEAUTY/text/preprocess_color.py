import pandas as pd
import numpy as np
import math
from tqdm import tqdm 
import matplotlib.pyplot as plt 

data = pd.read_csv("beauty_data_info_train_competition.csv")

result = []

rows = np.asarray(data.iloc[:, :])
for i in tqdm(range(0,len(rows))):
    color = rows[i][5]
    if not math.isnan(color):
        result.append(rows[i])

pd.DataFrame(result).to_csv("mixed_color.csv", index=False, header=list(data.columns))

data = pd.read_csv("mixed_color.csv")
color = data['Colour_group'].value_counts()
blist = list(color.index)
btimes = {blist[i]:i for i in range(len(blist))}

b_numerical = [btimes[e] for e in data['Colour_group']]
data['color_numerical'] = b_numerical 

data.to_csv("numerical_colors.csv", index=False, header=list(data.columns))
