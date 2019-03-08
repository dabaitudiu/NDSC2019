from keras.preprocessing import image 
import matplotlib.pyplot as plt 
import numpy as np 

def convert_img(img_path):
    try:
        img = image.load_img(img_path, target_size=(128,128))
        x = image.img_to_array(img)
        return x
    except:
        print("Error at image: ",img_path)