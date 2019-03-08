from keras.applications import VGG16,InceptionV3
import pandas as pd
from keras import layers 
from keras import optimizers 
from keras import models

def create_model():
    conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=(128,128,3))
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