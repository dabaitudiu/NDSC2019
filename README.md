# NDSC2019

### Descriptions:

#### Common Files/Folders Declaration:
- ```config```: config files
- ```tools.c_img_array```: 传入图片url，返回scale后的image array
- ```tools.DataGenerator```: train data 生成器， 避免out of memory
- ```tools.post_process```: 训练后print train & val's loss & accuracy
- ```models/```: CNN模型

#### LABEL-Benefits：
- 1.[data_manipulations/mixed_data_extraction_BENEFITS.py](www.github.com) 去除所有benefits=nan的entries， 生成 "mixed_benefits.csv" 文件
- 2.[data_manipulations/pre_generation_images_BENEFITS.py](www.github.com) 根据 "mixed_benefits.csv"，生成增强rare分类的图片，存入 "reinforced_images/"文件夹，生成信息汇总 "Reinforce_mixed_benefits.csv"
- 3.[train_benefits.py](www.github.com) 文件内定义class DataGenerator,以fit_generator的形式train
- 4.Best result: [best_model_05.hdf5](www.github.com) Epcoh 5 in 20. Accuracy = 65% on self-splitted validation set.  
- 5.(Need to be done) 用model对val set预测结果

#### LABEL-Brands：
- 1. [data_manipulations/preprocess_BRAND.py]() 去除所有brand=nan的entries，重新分配1-300的labels
- 2. [process_general.py]()
```python
def manipulate(TAG_NAME,BATCH_SIZE,FILEPATH,LABEL,NUM_CLASSES,MODEL,LAYERS,EPOCHS):
"""
参数说明：
TAG_NAME: 本次训练的nick name，用于生成的文件名
BATCH_SIZE: batch_size
FILE_PATH: 预处理好的mixed_class.csv 如numerical_brands.csv
LABEL: 本次要处理的LABEL的column index，e.g. benfitis->4, brands->5, (预处理后是8th column)
NUM_CLASS: 本次要处理的LABEL的class number
LAYERS:传入NN的网络层数，默认dropout=0.5, 如要修改，visit models/ 文件夹
EPOCHS: 训练的轮数
"""
```
- 3. [train_brand.py]() 训练brand
- 4. (Unifinshed) Due to encapsulation in LABEL-Brands, some reference in LABEL-Benefits'codes should be changed.

