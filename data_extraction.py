import pandas as pd
import numpy as np
import math
from tqdm import tqdm 

data = pd.read_csv("beauty_data_info_train_competition.csv")

color_values = list(data['Benefits'].value_counts().index.astype(int))
sub_classes = {}
for e in color_values:
    sub_classes[e] = []

rows = np.asarray(data.iloc[:, :])
for i in tqdm(range(0,len(rows))):
    color = rows[i][3]
    if not math.isnan(color):
        sub_classes[int(color)].append(rows[i])

for i in range(6):
    new_df = pd.DataFrame(sub_classes[i]).to_csv("benefit_{}.csv".format(i), index=False, header=0)
    