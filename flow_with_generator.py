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

BATCH_SIZE = 64
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data = pd.read_csv("Reinforced_mixed_benefits.csv")
rows = np.asarray(data.iloc[:, :])
print("Start shuffling.")
np.random.shuffle(rows)
print("Shuffling finished.")

train_data = rows[:int(0.6*(len(rows)))]
val_data = rows[int(0.6*(len(rows))):int(0.8*(len(rows)))]
test_data = rows[int(0.8*(len(rows))):]

pd.DataFrame(test_data).to_csv("test_data.csv",header=0,index=False)

class DataGenerator(Sequence):

    def __init__(self, datas, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)
        # print("_getitem: ",X.shape,y.shape)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            #x_train数据
            feautres = cia.convert_img(data[2])
            images.append(feautres)
            #y_train数据 
            labels.append(to_categorical(int(data[3]),num_classes=7))
        # print("data_generation: ",np.array(images).shape,np.array(labels).shape)
        return np.array(images), np.array(labels)

train_generator = DataGenerator(datas=train_data,batch_size=BATCH_SIZE)
val_generator = DataGenerator(datas=val_data,batch_size=BATCH_SIZE)

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

steps_per_epoch = math.ceil(len(data) / BATCH_SIZE)

history = model.fit_generator(train_generator,
        steps_per_epoch=steps_per_epoch,epochs=10,validation_data=val_generator)

model.save("generator.h5")

# history = model.fit(X_train, y_train, epochs=5, batch_size=64,validation_data=(X_val,y_val))
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print('Test accuracy:', test_acc)
# predictions = model.predict(X_test)

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
