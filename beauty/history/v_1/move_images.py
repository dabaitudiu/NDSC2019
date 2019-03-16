import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from PIL import Image
from shutil import copy
from tqdm import tqdm 

for k in range(7):
    print("Class ", k)
    data = pd.read_csv("benefit_{}.csv".format(k))
    rows = np.asarray(data.iloc[:, :])
    for i in tqdm(range(0,len(data))):
        image_url = rows[i][2]
        # img=mpimg.imread(image_url)
        # imgplot = plt.imshow(img)
        # plt.show()
        try:
            copy(image_url,"Benefits_bk/b{}/".format(k))
        except:
            print(image_url,"cannot be found.")


