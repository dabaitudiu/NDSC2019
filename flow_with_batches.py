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

SPLIT_TIMES = 10
SPLIT_POINT = 0.2

data = pd.read_csv("Reinforced_mixed_benefits.csv")
rows = np.asarray(data.iloc[:, :])
print("Start shuffling.")
np.random.shuffle(rows)
print("Shuffling finished.")

PARA_LEN = int(len(rows) / SPLIT_TIMES)

train_data = rows[:int(0.6*(len(rows)))]
val_data = rows[int(0.6*(len(rows))):int(0.8*(len(rows)))]
test_data = rows[int(0.8*(len(rows))):]

pd.DataFrame(test_data).to_csv("test_data_batches.csv",header=0,index=False)

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

def split_batches(whole_data,times):
    start = 0
    num_per_batch = int(len(whole_data) / times)
    end = num_per_batch
    batches = []
    for i in range(times):
        batches.append(whole_data[start:end]) 
        start += num_per_batch
        end += num_per_batch
    return batches
    
total_batches = split_batches(rows,SPLIT_TIMES)

def train_separate_batch(i)
    print("Ok. Train Batch {}, First load previous model, then continue training.".format(i))
    current_data = total_batches[i]
    split_point = int(0.8 * PARA_LEN)
    x = []
    y = []
    for j in range(len(current_data)):
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
        model = load_model("separate_batch{}.h5".format(i))

    history = model.fit(x_train, y_train, epochs=5, batch_size=64,validation_data=(x_val,y_val))
    model.save("separate_batch{}.h5".format(i))

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
    plt.savefig("Training_and_validation_loss_batch{}.png".format(i))

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
    plt.savefig("Training_and_validation_accuracy_batch{}.png".format(i))

for i in range(SPLIT_TIMES):
    try:
        train_separate_batch(i)
    except:
        print("Error happens at batch {}".format(i))

# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]


# steps_per_epoch = math.ceil(len(data) / 32)

# history = model.fit_generator(train_generator,
#         steps_per_epoch=steps_per_epoch,epochs=20,validation_data=val_generator,callbacks=callbacks_list)

# model.save("batch1.h5")

# # history = model.fit(X_train, y_train, epochs=5, batch_size=64,validation_data=(X_val,y_val))
# # test_loss, test_acc = model.evaluate(X_test, y_test)
# # print('Test accuracy:', test_acc)
# # predictions = model.predict(X_test)

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()
# plt.savefig("Training_and_validation_loss.png")

# plt.clf()

# acc = history.history['acc']
# val_acc = history.history['val_acc']

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()
# plt.savefig("Training_and_validation_accuracy.png")
