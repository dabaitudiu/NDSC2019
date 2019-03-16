from keras.models import Model 
from keras import layers 
from keras.applications import VGG16 
from keras import models
from keras import Input,layers
from keras.layers import Merge
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical
from keras.preprocessing import sequence 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import c_img_array as cia
import os 
import pickle 
import matplotlib.pyplot as plt 

max_features = 10000
word2vec_model = Word2Vec.load('word2vec/word2vec.100d.mfreq5.model')
UNK_WORD = len(word2vec_model.stoi)
UNK = UNK_WORD + 1

finished=1
if finished:
    stop_word = pickle.load(open('benefits_word_stopword_size300.pkl', 'rb'))
else:
    word_dict = construct_dict(data)
    print(len(word_dict))
    stop_word = [e for e in word_dict if word_dict[e] <= 2]
    print(len(stop_word))
    pickle.dump(set(stop_word), open('benefits_word_stopword_size300.pkl', 'wb'))

def combined_model(dense_layers,output_dim):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(128,128,3))
    image_model = models.Sequential()
    image_model.add(conv_base)
    image_model.add(layers.Flatten())
    image_model.add(layers.Dense(2048, activation='relu'))
    image_model.add(layers.Dropout(0.5))
    image_model.add(layers.Dense(1024, activation='relu'))
    image_model.add(layers.Dropout(0.5))
    image_model.add(layers.Dense(512, activation='relu'))
    # image_model.add(layers.Dropout(0.5))
    # image_model.add(layers.Dense(256, activation='relu'))

    # text_model = models.Sequential()
    # text_model.add(layers.Embedding(max_features, 200))
    # text_model.add(layers.LSTM(200))

    text_model = models.Sequential()
    text_model.add(layers.Embedding(UNK_WORD + 1, 100, input_length=20))
    text_model.add(layers.Convolution1D(256, 3, padding='same'))
    text_model.add(layers.MaxPool1D(3,3,padding='same'))
    text_model.add(layers.Convolution1D(128, 3, padding='same'))
    text_model.add(layers.MaxPool1D(3,3,padding='same'))
    text_model.add(layers.Convolution1D(64, 3, padding='same'))
    text_model.add(layers.Flatten())
    # text_model.add(layers.Dropout(0.5))
    # text_model.add(layers.BatchNormalization()) # (批)规范化层
    # text_model.add(layers.Dense(256,activation='relu'))
    # text_model.add(layers.Dropout(0.5))
    # text_model.add(layers.Dense(7,activation='softmax'))

    model = models.Sequential() 
    model.add(Merge([image_model,text_model], mode='concat'))
    model.add(layers.BatchNormalization())
    for item in dense_layers:
        model.add(layers.Dense(item, activation='relu'))
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_dim, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model 


data = pd.read_csv("Reinforced_mixed_benefits_1000.csv")
rows = np.asarray(data.iloc[:, :])
print("Start shuffling.")
np.random.shuffle(rows)
print("Shuffling finished.")

PARA_LEN = int(len(rows))

titles = np.asarray(data['title'])

def word2id(max_len):
    train_x = []
    for d in tqdm(titles):
        d = d.split()
        line = []
        for token in d:
            if token in stop_word or token == '' or token == ' ':
                continue
            if token in word2vec_model.stoi:
                line.append(word2vec_model.stoi[token])
            else:
                line.append(UNK)
        train_x.append(line[:max_len])
    return train_x


def train_full_batch():
    print("Ok. Train Full Batch, start training.")
    current_data = rows
    split_point = int(0.8 * PARA_LEN)
    x = []
    y = []
    for j in tqdm(range(len(current_data))):
        features = cia.convert_img(current_data[j][2])
        label = to_categorical(int(current_data[j][3]),num_classes=7)[0]
        x.append(features)
        y.append(label)

    x = np.array(x)
    y = np.array(y)

    print("Array conversion finished.")

    x_train = x[:split_point]
    y_train = y[:split_point]
    x_val = x[split_point:]
    y_val = y[split_point:]

    train_x_word = word2id(max_len=20)

    finished=0
    if finished:
        train_x_word = pickle.load(open('r-benefits-text-features', 'rb'))
    else:
        print("Text Features not extracted, start extracting...")
        train_x_word = word2id(max_len=20)
        pickle.dump(train_x_word, open('r-benefits-text-features', 'wb'))



    print("start padding sequences")
    x_nlp = sequence.pad_sequences(train_x_word, maxlen=20, dtype='int32', padding='post', truncating='post', value=UNK)
    print("padding finished.")

    x_nlp_train=x_nlp[:split_point]
    x_nlp_val=x_nlp[split_point:]

    model = combined_model([256,128,32],7)
    history = model.fit([x_train,x_nlp_train], y_train, epochs=5, batch_size=64,validation_data=([x_val,x_nlp_val],y_val))
    model.save("full_batch.h5")

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
    plt.savefig("Training_and_validation_loss_full_batch{}.png")

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
    plt.savefig("Training_and_validation_accuracy_full_batch{}.png")

train_full_batch()