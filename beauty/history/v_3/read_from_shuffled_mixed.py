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
from keras.utils import Sequence
import math
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

SPLIT_TIMES = 5
CURRENT_ROUND = 0

fnames = []
for i in range(SPLIT_TIMES):
    fname = "Shuffled_benefits_{}_of_{}.csv".format(i,i+1)
    fnames.append(fname)

data = pd.read_csv(fnames[CURRENT_ROUND])
rows = np.asarray(data.iloc[:, :])


def training_model(model_name):
    if model_name == "VGG16":
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(7, activation='softmax'))
        model.summary()

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                loss='categorical_crossentropy',
                metrics=['acc'])
        return model 

def train_separate_batch(i):
    print("Ok. Train Batch {}, First load previous model, then continue training.".format(i))
    current_data = total_batches[i]
    split_point = int(0.8 * PARA_LEN)
    x = []
    y = []
    for j in tqdm(range(len(current_data))):
        features = cia.convert_img(current_data[j][2])
        label = to_categorical(int(current_data[j][3]),num_classes=7)
        x.append(features)
        y.append(label)
    x_train = x[:split_point]
    y_train = y[:split_point]
    x_val = x[split_point:]
    y_val = y[split_point:]

    if (i == 0):
        model = training_model("VGG16")
    else:
        model = load_model("separate_batch_{}_of_{}.h5".format(i,SPLIT_TIMES))

    history = model.fit(x_train, y_train, epochs=5, batch_size=64,validation_data=(x_val,y_val))
    model.save("separate_batch_{}_of_{}.h5".format(i,SPLIT_TIMES))

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
    plt.savefig("Loss_batch_{}_of_{}.png".format(i,SPLIT_TIMES))

    plt.clf()

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    plt.savefig("Accuracy_batch_{}_of_{}.png".format(i,SPLIT_TIMES))

# for i in range(SPLIT_TIMES):
#     try:
#         train_separate_batch(i)
#     except:
#         print("Error happens at batch {}".format(i))

train_separate_batch(0)