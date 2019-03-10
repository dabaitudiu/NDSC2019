import pandas as pd
import numpy as np
import math
from tqdm import tqdm 

data = pd.read_csv("beauty_data_info_train_competition.csv")

result = []

rows = np.asarray(data.iloc[:, :])
for i in tqdm(range(0,len(rows))):
    color = rows[i][3]
    if not math.isnan(color):
        result.append(rows[i])


pd.DataFrame(result).to_csv("mixed_benefits.csv", index=False, header=data.columns)
    