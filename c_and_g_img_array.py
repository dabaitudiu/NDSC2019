from keras.preprocessing import image 
import matplotlib.pyplot as plt 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt 

def convert_and_generage_img(img_path,times):
    try:
        result = []

        datagen = ImageDataGenerator(rotation_range=50, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.2,
        zoom_range=0.3, horizontal_flip=True,fill_mode='nearest')

        img = image.load_img(img_path, target_size=(128,128))
        x = image.img_to_array(img)
        result.append(x)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            result.append(batch[0])
            # plt.figure(i)
            # imgplot = plt.imshow(image.array_to_img(batch[0]))
            i += 1
            if i % times == 0:
                break
        return result
    except:
        print("Error at image: ",img_path)

model.save('NDSC2019_v1.h5') 
