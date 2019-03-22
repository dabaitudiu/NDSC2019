import pandas as pd 
from tqdm import tqdm 
import numpy as np 
from gensim.models.word2vec import Word2Vec
from keras.models import Model 
from keras import layers 
from keras.applications import VGG16 
from keras import models
from keras import Input,layers
# from keras.layers import Merge
from keras.preprocessing import sequence 
import pickle 
from keras.utils import to_categorical
from models import SimpleCNN
from keras.callbacks import ModelCheckpoint

filename1 = 'fashion_data_info_train_competition.csv'
filename2 = 'fashion_numerical_Pattern.csv'
BigLabelName = "Fashion"
SmallLabelName = "Pattern"
label_Classes = 20
dict_finished = 0

# def train_w2v_model(min_freq=5, size=100):
#     words = []
#     corpus = data['title']
#     for e in tqdm(corpus):
#         words.append([i for i in e.strip().split() if i])
#     print('Traning set corpus:', len(corpus))
#     print('Total Length: ', len(words))
#     model = Word2Vec(words, size=size, window=5, min_count=min_freq)
#     model.itos = {}
#     model.stoi = {}
#     model.embedding = {}
    
#     print('Save model...')
#     for k in tqdm(model.wv.vocab.keys()):
#         model.itos[model.wv.vocab[k].index] = k
#         model.stoi[k] = model.wv.vocab[k].index
#         model.embedding[model.wv.vocab[k].index] = model.wv[k]

#     model.save('word2vec/{}_word2vec.{}d.mfreq{}.model'.format(BigLabelName, size, min_freq))
#     return model

# if not dict_finished:
#     data = pd.read_csv(filename1)
#     model = train_w2v_model(size=100)

data = pd.read_csv(filename2)

# 处理低频词
finished = 1
def construct_dict(df):
    word_dict = {}
    corput = df.title
    for line in tqdm(corput):
        for e in line.strip().split():
            word_dict[e] = word_dict.get(e, 0) + 1
    return word_dict

if finished:
    stop_word = pickle.load(open('{}_benefits_word_stopword_size300.pkl'.format(BigLabelName), 'rb'))
else:
    word_dict = construct_dict(data)
    print(len(word_dict))
    stop_word = [e for e in word_dict if word_dict[e] <= 2]
    print(len(stop_word))
    pickle.dump(set(stop_word), open('{}_benefits_word_stopword_size300.pkl'.format(BigLabelName), 'wb'))

word2vec_model = Word2Vec.load('word2vec/{}_word2vec.100d.mfreq5.model'.format(BigLabelName))
UNK_WORD = len(word2vec_model.stoi)
UNK = UNK_WORD + 1

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

train_x_word = word2id(max_len=20)

model = models.Sequential()
model.add(layers.Embedding(UNK_WORD + 1, 100, input_length=20))
model.add(layers.Convolution1D(256, 3, padding='same'))
model.add(layers.MaxPool1D(3,3,padding='same'))
model.add(layers.Convolution1D(128, 3, padding='same'))
model.add(layers.MaxPool1D(3,3,padding='same'))
model.add(layers.Convolution1D(64, 3, padding='same'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization()) # (批)规范化层
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(label_Classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

print("To categorical")
tmp_y = np.asarray(data['numerical_{}'.format(SmallLabelName)]).astype(int)
y = []
for i in range(len(tmp_y)):
    y.append(to_categorical(tmp_y[i],num_classes=label_Classes))
y = np.array(y)

print("start padding sequences")
x = sequence.pad_sequences(train_x_word, maxlen=20, dtype='int32', padding='post', truncating='post', value=UNK)
print("padding finished.")

x_train = x[:int(0.8 * len(x))]
x_test = x[int(0.8 * len(x)):]
y_train = y[:int(0.8 * len(y))]
y_test = y[int(0.8 * len(x)):]

checkpoint = ModelCheckpoint("saved_models/{}_{}_best_model.hdf5".format(BigLabelName,SmallLabelName), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
callback_lists = [checkpoint] 

model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_test, y_test),callbacks=callback_lists)
