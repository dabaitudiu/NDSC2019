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


data = pd.read_csv("mixed_benefits_small.csv")
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

print("-"*30,"Start train-test splitting.","-"*30)
X_train_val, X_test, y_train_val, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
print("-"*30,"Splitting finished.","-"*30)

print("-"*30,"Start train-val splitting.","-"*30)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=33)
print("-"*30,"Splitting finished.","-"*30)

print("-"*30,"Start creating model.","-"*30)
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


# model = multi_gpu_model(model, 2)
model.summary()

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
        loss='categorical_crossentropy',
        metrics=['acc'])

# model.fit_generator(data_generator(batch_size=batch_size),
#         samples_per_epoch=samples_per_epoch,epochs=20,validation_data=(X_test, y_test))

history = model.fit(X_train, y_train, epochs=5, batch_size=64,validation_data=(X_val,y_val))
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(X_test)

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
plt.savefig("Training_and_validation_loss.png")

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
plt.savefig("Training_and_validation_accuracy.png")
