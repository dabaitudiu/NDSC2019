import numpy as np 
import math 
from keras.utils import Sequence
import c_img_array as cia
from keras.utils import to_categorical


class DataGenerator(Sequence):

    def __init__(self, datas, batch_size=64, shuffle=True,label=3,num_classes=7):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle
        self.label = label 
        self.num_classes = num_classes
        # print("batch size is ", batch_size)

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
            labels.append(to_categorical(int(data[self.label]),num_classes=self.num_classes))
        # print("data_generation: ",np.array(images).shape,np.array(labels).shape)
        return np.array(images), np.array(labels)