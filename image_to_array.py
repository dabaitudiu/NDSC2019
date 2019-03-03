from keras.preprocessing import image 
import os
import matplotlib.pyplot as plt 
import pickle
from tqdm import tqdm 
import numpy as np 

# images path
train_dir = "Benefits_bk/b1/"

fnames = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir)]
result = []

for k in tqdm(range(30000,len(fnames))):
    img_path = fnames[k]
    try:
        img = image.load_img(img_path, target_size=(128,128))
        x = image.img_to_array(img)
        result.append(x)
    except:
        print("Error at image: ",img_path)

pickle.dump(result, open('benefit1-d.pkl', 'wb'))