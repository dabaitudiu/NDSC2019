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
import tools.c_img_array as cia
import tools.post_process
import os 
import pickle 
import matplotlib.pyplot as plt 
from keras.utils import Sequence
import math

class DataGenerator(Sequence):
    def __init__(self, datas1, datas2, batch_size=BATCH_SIZE, shuffle=True):
        self.batch_size = batch_size
        self.datas1 = datas1
        self.datas2 = datas2
        self.indexes = np.arange(len(self.datas1))
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.datas1) / float(self.batch_size))

    def __getitem__(self, index):
        
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_datas = [self.datas1[k] for k in batch_indexs]
        batch_text_datas = [self.datas2[k] for k in batch_indexs]

        x_image, y = self.data_generation(batch_datas)
        x_text = self.text_data_generation(batch_text_datas)

        return [x_image,x_text], y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        for i, data in enumerate(batch_datas):
            feautres = cia.convert_img(data[2])
            images.append(feautres)
            labels.append(to_categorical(int(data[3]),num_classes=7)[0])
        return np.array(images), np.array(labels)

    def text_data_generation(self, batch_datas):
        texts = []
        for i, data in enumerate(batch_datas):
            texts.append(data)
        return np.array(texts)


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

    text_model = models.Sequential()
    text_model.add(layers.Embedding(UNK_WORD + 1, 100, input_length=20))
    text_model.add(layers.Convolution1D(256, 3, padding='same'))
    text_model.add(layers.MaxPool1D(3,3,padding='same'))
    text_model.add(layers.Convolution1D(128, 3, padding='same'))
    text_model.add(layers.MaxPool1D(3,3,padding='same'))
    text_model.add(layers.Convolution1D(64, 3, padding='same'))
    text_model.add(layers.Flatten())

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

    x_image_train = rows[:int(0.8*(len(rows)))]
    x_image_val = rows[int(0.8*(len(rows))):]

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

    train_generator = DataGenerator(datas1=x_image_train, datas2=x_nlp_train,batch_size=BATCH_SIZE)
    val_generator = DataGenerator(datas1=x_image_val, datas2=x_nlp_val,batch_size=BATCH_SIZE)

    model = combined_model([256,128,32],7)
    steps_per_epoch = math.ceil(len(x_image_train) / BATCH_SIZE)
    validation_steps = math.ceil(len(x_image_val) / BATCH_SIZE)
    history = model.fit_generator(train_generator, epochs=5, steps_per_epoch=steps_per_epoch,validation_data=val_generator,validation_steps=validation_steps)
    model.save("full_batch.h5")

    # show loss and accuracy graphs.
    post_process(history)

if __name__ == "__main__":

    BATCH_SIZE=32
    max_features = 10000
    word2vec_model = Word2Vec.load('word2vec/word2vec.100d.mfreq5.model')
    UNK_WORD = len(word2vec_model.stoi)
    UNK = UNK_WORD + 1

    finished=1
    if finished:
        stop_word = pickle.load(open('benefits_word_stopword_size300.pkl', 'rb'))
    else:
        word_dict = construct_dict(data)
        stop_word = [e for e in word_dict if word_dict[e] <= 2]
        pickle.dump(set(stop_word), open('benefits_word_stopword_size300.pkl', 'wb'))

    data = pd.read_csv("Reinforced_mixed_benefits_1000.csv")
    rows = np.asarray(data.iloc[:, :])
    np.random.shuffle(rows)
    PARA_LEN = int(len(rows))
    titles = np.asarray(data['title'])

    train_full_batch()
