import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image 
from tqdm import tqdm 
import numpy as np 


data = pd.read_csv("mixed_benefits_1000.csv")
rows = np.asarray(data.iloc[:, :])

datagen = ImageDataGenerator(rotation_range=50, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.2,
        zoom_range=0.3, horizontal_flip=True,fill_mode='nearest')

image_index = 0
image_id = 0

result = []

for i in tqdm(range(len(rows))):
    result.append(rows[i])
    benefit_class = int(rows[i][3])
    item_description = rows[i][1]
    img_path = rows[i][2]
    if (benefit_class == 2) or (benefit_class == 5):
        times = 10 if benefit_class == 2 else 5
        try:
            img = image.load_img(img_path, target_size=(128,128))
            x = image.img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                tmp = image.array_to_img(batch[0])
                new_url = "reinforced_images/generated_{}.jpg".format(image_index)
                tmp.save(new_url,"JPEG")
                new_entry = [image_id,item_description,new_url,benefit_class,0,0,0,0]
                result.append(new_entry)
                image_index += 1
                image_id += 1
                i += 1
                if i % times == 0:
                    break
        except:
            print("Error at image: ",img_path)

pd.DataFrame(result).to_csv("Reinforce_mixed_benefits.csv",index=False, header=0)
