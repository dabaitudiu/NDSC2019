import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Input, Embedding, Conv1D, Conv2D, Flatten, Concatenate, Dense, Dropout, Softmax
from keras.models import Model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import pandas as pd
from params import CSV_TRAIN_PATH, MODEL_SAVE_PATH, IMG_DATA_PATH, LABEL_TYPES, LABEL_DIMS, \
    VALIDATION_SPLIT, IMG_SIZE, \
    EMBEDDING_DIM, LR_SCHEDULER_DURATION, LR_SCHEDULER_END, LR_START, LR_END, BATCH_SIZE, N_EPOCHS
from lr_scheduler import CosineLRScheduler
import os


class DataGenerator:
    def __init__(self, df, tokenize_func, img_transform_func=None, data_dir=IMG_DATA_PATH, batch_size=BATCH_SIZE):
        self.df = df.sample(frac=1.)
        self.data_dir = data_dir
        self.batch_size = batch_size

        txts = self.df['title'].astype(str).values
        self.X_txt = tokenize_func(txts)
        self.transform_img = img_transform_func
        self.targets = [self.df[l].fillna(LABEL_DIMS[i]).astype(int).values for i, l in enumerate(LABEL_TYPES)]

        self.batch_start = 0
        self.batch_end = BATCH_SIZE
        self.dataset_size = len(self.df)

    def batch_generator(self):
        while True:
            imgs = []
            for i in range(self.batch_start, self.batch_end):
                if i >= self.dataset_size:
                    i -= self.dataset_size
                img = load_img(os.path.join(IMG_DATA_PATH, self.df.iloc[i]['image_path']), target_size=IMG_SIZE)
                img = img_to_array(img)
                if self.transform_img is not None:
                    img = self.transform_img(img)
                imgs.append(img)
            imgs_batch = np.asarray(imgs)

            if self.batch_end > self.dataset_size:
                self.batch_end -= self.dataset_size
                txts_batch = np.concatenate([self.X_txt[self.batch_start:], self.X_txt[:self.batch_end]])
                targets_batch = [np.concatenate([Y[self.batch_start:], Y[:self.batch_end]])
                                 for Y in self.targets]
            else:
                txts_batch = self.X_txt[self.batch_start: self.batch_end]
                targets_batch = [Y[self.batch_start: self.batch_end] for Y in self.targets]

            self.batch_start = self.batch_end
            self.batch_end += self.batch_size

            yield ([txts_batch, imgs_batch], targets_batch)


class MobileModel:
    def __init__(self):
        df = pd.read_csv(CSV_TRAIN_PATH)
        df = df.sample(frac=1.)

        self.maxlen = max([len(s.split()) for s in df['title'].values])

        txts = df['title'].astype(str).values
        tokenizer = Tokenizer(lower=True)
        tokenizer.fit_on_texts(txts)
        self.n_words = len(tokenizer.word_index)
        print(f'Found {self.n_words} unique tokens.')

        split_idx = int(len(df) * VALIDATION_SPLIT)
        df_train = df[split_idx:]
        df_valid = df[:split_idx]
        self.train_size = len(df_train)
        self.valid_size = len(df_valid)

        tokenize_func = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=self.maxlen)
        img_transform_func = ImageDataGenerator(rotation_range=20,
                                                width_shift_range=.2,
                                                height_shift_range=.2,
                                                shear_range=.1,
                                                zoom_range=.2,
                                                rescale=1. / 255).random_transform

        self.datagen_train = DataGenerator(df_train,
                                           tokenize_func=tokenize_func,
                                           img_transform_func=img_transform_func)
        self.datagen_valid = DataGenerator(df_valid,
                                           tokenize_func=tokenize_func)

        self.model = self.build_model()

    def build_model(self):
        inputs_txt = Input(shape=(self.maxlen,))
        x_txt = Embedding(self.n_words, EMBEDDING_DIM, trainable=True)(inputs_txt)
        x_txt = Conv1D(filters=32, kernel_size=4, activation='elu')(x_txt)
        x_txt = Conv1D(filters=32, kernel_size=2, activation='elu')(x_txt)
        x_txt = Conv1D(filters=16, kernel_size=2, activation='elu')(x_txt)
        x_txt = Conv1D(filters=16, kernel_size=2, strides=2, activation='elu')(x_txt)
        x_txt = Flatten()(x_txt)
        x_txt = Dense(64, activation='elu')(x_txt)
        x_txt = Dropout(.2)(x_txt)

        inputs_img = Input(shape=IMG_SIZE+(3,))
        x_img = Conv2D(filters=32, kernel_size=4, activation='elu')(inputs_img)
        x_img = Conv2D(filters=32, kernel_size=4, activation='elu')(x_img)
        x_img = Conv2D(filters=32, kernel_size=4, strides=2, activation='elu')(x_img)
        x_img = Conv2D(filters=16, kernel_size=2, activation='elu')(x_img)
        x_img = Conv2D(filters=16, kernel_size=2, strides=2, activation='elu')(x_img)
        x_img = Conv2D(filters=16, kernel_size=2, activation='elu')(x_img)
        x_img = Conv2D(filters=16, kernel_size=2, strides=2, activation='elu')(x_img)
        x_img = Conv2D(filters=16, kernel_size=2, activation='elu')(x_img)
        x_img = Conv2D(filters=16, kernel_size=2, strides=2, activation='elu')(x_img)
        x_img = Flatten()(x_img)
        x_img = Dense(64, activation='elu')(x_img)
        x_img = Dropout(.2)(x_img)

        x = Concatenate()([x_txt, x_img])

        x = Dense(64)(x)
        x = Dropout(.5)(x)
        outputs = [Dense(dim + 1, name='dense_' + LABEL_TYPES[i].replace(' ', ''))(x) for i, dim in
                   enumerate(LABEL_DIMS)]
        outputs = [Softmax(name=LABEL_TYPES[i].replace(' ', ''))(y) for i, y in enumerate(outputs)]

        model = Model(inputs=[inputs_txt, inputs_img],
                      outputs=outputs,
                      name='mobile_model')

        losses = ['sparse_categorical_crossentropy'] * len(LABEL_TYPES)
        loss_wts = [1.] * len(LABEL_TYPES)

        model.compile(loss=losses, loss_weights=loss_wts,
                      optimizer='Nadam',
                      metrics=['acc'])

        print(model.summary())

        return model

    def train_model(self):
        saver = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)
        lr_scheduler = CosineLRScheduler(LR_SCHEDULER_DURATION, LR_SCHEDULER_END, LR_START, LR_END)
        lr_scheduler = LearningRateScheduler(lr_scheduler.update, verbose=1)
        self.model.fit_generator(self.datagen_train.batch_generator(),
                                 steps_per_epoch=self.train_size // BATCH_SIZE,
                                 epochs=N_EPOCHS,
                                 verbose=2,
                                 callbacks=[saver, lr_scheduler],
                                 validation_data=self.datagen_valid.batch_generator(),
                                 validation_steps=self.valid_size // BATCH_SIZE)


if __name__ == "__main__":
    model = MobileModel()
    model.train_model()