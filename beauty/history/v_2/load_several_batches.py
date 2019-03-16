from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt 
import pickle
from tqdm import tqdm 
import numpy as np 
import c_img_array as cia
import c_and_g_img_array as cgia
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications import VGG16 
import pandas as pd
from keras import layers 
from keras import optimizers 
from keras import models
from keras.utils import multi_gpu_model
from keras.models import load_model


data = pd.read_csv("20000_40000.csv")
rows = np.asarray(data.iloc[:, :])

data_x = []
data_y = []


for i in tqdm(range(0,len(data))):
    image_class = int(rows[i][3])
    image_url = rows[i][2]
    if image_class == 2:
        tmp_result = cgia.convert_and_generage_img(image_url,10)
        for item in tmp_result:
            data_x.append(item)
        for j in range(11):
            data_y.append(to_categorical(2, num_classes=7))
    elif image_class == 5:
        tmp_result = cgia.convert_and_generage_img(image_url,5)
        for item in tmp_result:
            data_x.append(item)
        for j in range(6):
            data_y.append(to_categorical(5, num_classes=7))
    else:
        data_x.append(cia.convert_img(image_url))
        data_y.append(to_categorical(image_class,num_classes=7))

print("-"*30,"Start converting data_x to array","-"*30)
data_x = np.array(data_x)
print("-"*30,"Start converting data_y to array.","-"*30)
data_y = np.array(data_y)
print("-"*30,"Conversion to array finished.","-"*30)

print("-"*30,"Start creating model.","-"*30)

model = load_model("benefits_b1.h5")

history = model.fit(data_x,data_y, epochs=20, batch_size=64)
model.save("benefits_b2.h5")