# NDSC2019

### Usage:

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
- generator憋不出来，改用大内存了
- 不单独把图片分类了，过滤所有class=NaN的entries, 遍历mixed_benefits.csv, 检测到属于稀有类直接reinforce,结果append到data里
- 集合在```flow_from_csv.py```中

**2019/3/5:**:
- generator 憋出来了, ```flow_with_generator.py```
- 提前生成增强图片并将信息存储到csv中,```pre_generation_images```
- generator batch_size=32时准确率有点堪忧，=64时达到了70%以上。
- 作为参考，也写了读入全部数据，然后分段save load训练的方法 ```flow_with_batches.py```
