import pickle 
import config
import numpy as np
import math
from keras import models
from keras import layers 
from keras import optimizers
from keras.applications import VGG16 
import matplotlib.pyplot as plt 

benefit_1,benefit_2,benefit_3,benefit_4,benefit_5,benefit_6 = config.initialize()
total_length = len(benefit_1) + len(benefit_2) + len(benefit_3) + len(benefit_4) + len(benefit_5) + len(benefit_6)

# allocate start indexes.
b1_start_index = 0
b2_start_index = len(benefit_1)
b3_start_index = len(benefit_2) + b2_start_index
b4_start_index = len(benefit_3) + b3_start_index
b5_start_index = len(benefit_4) + b4_start_index
b6_start_index = len(benefit_5) + b5_start_index

idx = np.arange(total_length)
np.random.shuffle(idx)

train_val_split = int(total_length * 0.6)
val_test_split = train_val_split + int(total_length * 0.2)

train_index = idx[:train_val_split]
val_index = idx[train_val_split:val_test_split]
test_index = idx[val_test_split:]

lead_index = 0

def data_distributor(current_index):
    if (current_index < b2_start_index): 
        return np.array(benefit_1[current_index]),1
    elif (current_index < b3_start_index):
        return np.array(benefit_2[current_index]),2
    elif (current_index < b4_start_index):
        return np.array(benefit_3[current_index]),3
    elif (current_index < b5_start_index):
        return np.array(benefit_4[current_index]),4
    elif (current_index < b6_start_index):
        return np.array(benefit_5[current_index]),5
    else: 
        return np.array(benefit_6[current_index]),6


# data generator
def data_generator(batch_size):
    while 1:
        cnt = 0
        X =[]
        Y =[]
        for i in range(len(train_index)):
            current_index = idx[lead_index]
            lead_index += 1
            x,y = data_distributor(current_index)
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []

def data_collector(start_index,end_index):
    X =[]
    Y =[]
    for i in range(end_index - start_index):
        current_index = idx[start_index]
        start_index += 1
        x,y = data_distributor(start_index)
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)
    
print("Start train data collection. ")
x_train,y_train = data_collector(0,train_val_split)
print("Start val data collection. ")
x_val,y_val = data_collector(train_val_split,val_test_split)
print("Start test data collection. ")
x_test,y_test = data_collector(test_index,len(total_length))
print("Splitting Finished.")

# batch_size = 32
# samples_per_epoch = math.ceil(train_num/batch_size) * batch_size

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
model.add(layers.Dense(6, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
        loss='categorical_crossentropy',
        metrics=['acc'])

# model.fit_generator(data_generator(batch_size=batch_size),
#         samples_per_epoch=samples_per_epoch,epochs=20,validation_data=(X_test, y_test))

history = model.fit(x_train, y_train, epochs=20, batch_size=64,validation_data=(x_val,y_val))
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
predictions = model.predict(x_test)

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