from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt 
import pickle
from tqdm import tqdm 
import numpy as np 

"""
Parameters Specifications:
rotation_range: 角度值，表示图像随即旋转的角度范围
width_shift & height_shift: 图像在水平或垂直方向上的平移的范围（相对于总宽度或高度的比例)
shear_range: 随机错切变换的角度
zoom_range: 图像随机缩放的范围
horizontal_flip: 随机将一半图像水平翻转
fill_mode: 用于填充新创建像素的方法.
"""
datagen = ImageDataGenerator(rotation_range=50, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.2,
zoom_range=0.3, horizontal_flip=True,fill_mode='nearest')

# images path
train_dir = "Benefits_bk/b2/"

fnames = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir)]
result = []

for k in tqdm(range(2000,len(fnames))):
	img_path = fnames[k]
	img = image.load_img(img_path, target_size=(128,128))
	x = image.img_to_array(img)
	result.append(x)
	x = x.reshape((1,) + x.shape)

	i = 0
	for batch in datagen.flow(x, batch_size=1):
		result.append(batch[0])
		# plt.figure(i)
		# imgplot = plt.imshow(image.array_to_img(batch[0]))
		i += 1
		if i % 10 == 0:
			break

# print(np.array(result).shape)
pickle.dump(result, open('benefit2-c.pkl', 'wb'))