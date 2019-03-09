import pandas as pd
import numpy as np
import math
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import tools.c_img_array as cia 
from keras.utils import to_categorical
from tools.DataGenerator import DataGenerator
from tools.post_process import post_process
import config
from keras.callbacks import ModelCheckpoint

def manipulate(TAG_NAME,BATCH_SIZE,FILEPATH,LABEL,NUM_CLASSES,MODEL,LAYERS,EPOCHS):
    print("Parameters: Timestamp:{} Batch size:{} Model:{} Epochs:{}".format(TAG_NAME,BATCH_SIZE,MODEL,EPOCHS))
    data = pd.read_csv(FILEPATH)
    rows = np.asarray(data.iloc[:, :])
    print("Shuffling brands.")
    np.random.shuffle(rows)
    print("Shuffle finished.")
    layers_list = LAYERS

    train_data = rows[:int(0.8*(len(rows)))]
    val_data = rows[int(0.8*(len(rows))):]

    train_generator = DataGenerator(datas=train_data,batch_size=BATCH_SIZE,label=LABEL,num_classes=NUM_CLASSES)
    val_generator = DataGenerator(datas=val_data,batch_size=BATCH_SIZE,label=LABEL,num_classes=NUM_CLASSES)

    model = config.model(MODEL).create_model_with_layers(layers_list,NUM_CLASSES)
    steps_per_epoch = math.ceil(len(data) / BATCH_SIZE)

    checkpoint = ModelCheckpoint("saved_models/{}_best_model.hdf5".format(TAG_NAME), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callback_lists = [checkpoint] 

    history = model.fit_generator(train_generator,
            steps_per_epoch=steps_per_epoch,epochs=EPOCHS,validation_data=val_generator,callbacks=callback_lists)

    model.save("model_{}.h5".format(TAG_NAME))

    post_process(history,TAG_NAME)



    


