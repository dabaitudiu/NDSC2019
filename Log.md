**2019/3/3:**
```
convert_128.py - 图片转为128*128格式
data_extraction.py - 从信息csv中提取图片及其对应组别
image_to_array.py - 图片数很多的组，直接转化成array
image_generation.py - 图片很少的组，图像变换数据增强
config.py - 加载数据，预处理，parser设置等
VGG16.py - VGG16模型
```

**2019/3/4:**
- Useless log thus deleted at March 9

**2019/3/5:**:
- Useless log thus deleted at March 9
<div align=center><img width="300" height="250" alt="loss" src="https://github.com/dabaitudiu/NDSC2019/blob/master/info_pics/loss.png"/></div>
<div align=center><img width="300" height="250" alt="accuracy" src="https://github.com/dabaitudiu/NDSC2019/blob/master/info_pics/accuracy.png"/></div>

**2019/3/7:**:
add parsers, change codes as modes and classes. ```train_benefits.py```

**2019/3/8:**:
brand data preprocess, delete all nan，re-allocate 1-300's label

**2019/3/9:**:
Enhanced encapsulation, begin brand training.