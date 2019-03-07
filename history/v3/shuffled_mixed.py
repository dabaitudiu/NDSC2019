import numpy as np 
import pandas as pd

SPLIT_TIMES = 5

data = pd.read_csv("Reinforced_mixed_benefits.csv")
rows = np.asarray(data.iloc[:, :])

PARA_LEN = int(len(rows) / SPLIT_TIMES)

print("Start shuffling.")
np.random.shuffle(rows)
print("Shuffling finished.")

start = 0
end = PARA_LEN

for i in range(SPLIT_TIMES):
    new_data = rows[start:end]
    start += PARA_LEN
    end += PARA_LEN
    pd.DataFrame(rows).to_csv("Shuffled_benefits_{}_of_{}.csv".format(i,SPLIT_TIMES),index=False,header=0)
