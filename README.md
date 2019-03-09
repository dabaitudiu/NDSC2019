# NDSC2019

### Descriptions:

#### Common Files/Folders Declaration:
- ```config```: config files
- ```tools.c_img_array```: feed in images' url，return image_array after scaling
- ```tools.DataGenerator```: train data generator， avoid out of memory
- ```tools.post_process```: print train & val's loss & accuracy after training
- ```models/```: CNN models

#### LABEL-Benefits：
- 1. [mixed_data_extraction_BENEFITS.py](https://github.com/dabaitudiu/NDSC2019/blob/master/data_manipulations/data_extraction_BENEFITS.py) delete all entries where benefits=nan， generate file "mixed_benefits.csv" .
- 2. [pre_generation_images_BENEFITS.py](https://github.com/dabaitudiu/NDSC2019/blob/master/data_manipulations/pre_generation_images_BENEFITS.py) Based on "mixed_benefits.csv"，generate enhanced images in rare classes，save into "reinforced_images/" and generate "Reinforce_mixed_benefits.csv".
- 3. [train_benefits.py](https://github.com/dabaitudiu/NDSC2019/blob/master/train_beneifts.py) define class DataGenerator,train using fit_generator.
- 4. Best result: [best_model_05.hdf5](www.github.com) Epcoh 5 in 20. Accuracy = 65% on self-splitted validation set.  
- 5. (Need to be done) predict on val_set.

#### LABEL-Brands：
- 1. [data_manipulations/preprocess_BRAND.py](https://github.com/dabaitudiu/NDSC2019/blob/master/data_manipulations/preprocess_BRAND.py) delete all entries where brand=NaN，re-allocate 1-300's labels.
- 2. [process_general.py](https://github.com/dabaitudiu/NDSC2019/blob/master/process_general.py)
```python
def manipulate(TAG_NAME,BATCH_SIZE,FILEPATH,LABEL,NUM_CLASSES,MODEL,LAYERS,EPOCHS):
"""
Suppose LABEL is the target we want to focus this time.

TAG_NAME: nick name for this training. Used for generated_file_names
BATCH_SIZE: batch_size
FILE_PATH: preprocessed mixed_class.csv. e.g. numerical_brands.csv
LABEL: LABEL's column index，e.g. benfitis->4, brands->5, (8th column after preprocessing)
NUM_CLASS: LABEL's class number
LAYERS:layers in NN，default dropout=0.5, pls visit models/ if you want to change.
EPOCHS: training epochs
"""
```
- 3. [train_brand.py]() train brand
- 4. (Unifinshed) Due to encapsulation in LABEL-Brands, some reference in LABEL-Benefits'codes should be changed.

