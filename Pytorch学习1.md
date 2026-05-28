# Pytorch学习1

## 一、PyTorch加载数据初认识

### 1.Dataset：提供一种方式去获取数据以及label，把杂乱无章的数据变成模型看的懂的标准格式，它需要完成以下功能：

+ 获取每一个数据及其标签
+ 获得数据的总数

### 2.Dataloader：为模型运输各种数据

## 二、Dataset 代码示例

## 1.整理文件夹：

+ 按照一下格式整理文件夹

<img src="C:\Users\35641\AppData\Roaming\Typora\typora-user-images\image-20260527170659520.png" alt="image-20260527170659520" style="zoom:100%;" />

+ 把这个文件夹含有蚂蚁图片的文件夹整理后就是这样：

<img src="C:\Users\35641\AppData\Roaming\Typora\typora-user-images\image-20260527170421272.png" alt="image-20260527170421272" style="zoom:50%;" />



### 2.编写代码：

~~~python
from torch.utils.data import Dataset
import os
from PIL import Image
class MyDataset(Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir 
        self.label = label 
        # 拿到文件夹里所有图片名
        self.img_path = os.listdir(os.path.join(self.root_dir, self.label)) #把文件夹里所有的照片的文件名都读了出来，并且存在了一个列表里

    def __getitem__(self, item):
        # 拿到第 item 张图片的名字
        img_name = self.img_path[item]
        # 拼接完整路径：train/ant/xxx.jpg
        img_path = os.path.join(self.root_dir, self.label, img_name)
        # 打开图片为RGB模式
        img = Image.open(img_path).convert("RGB"）
        label = self.label
        return img, label

    def __len__(self):
        # 返回总图片数
        return len(self.img_path)

if __name__ == '__main__':
    root_dir = "练手数据集/train"
    label = "ants"

    dataset = MyDataset(root_dir, label)


    print("总数量：", len(dataset))
    img, label = dataset[0]  
    print("图片形状："；, img.size)
    print("标签：", label)
~~~

## 三、tensorboard中writer.add_scalar

### 1.什么是tensorboard？

深度学习中的可视化工具，专门用来画曲线的。

### 2.代码示例：

~~~python
from torch.utils.tensorboard import SummaryWriter #导入tensorboard工具
writer = SummaryWriter("logs") #创建一个writer，把生成的东西放在一个交”logs“的文件夹
for i in range(10):
    writer.add_scalar("y=x",i,i)
    #tag：图表的名字，”y=x"
    #scalar_value：y轴的值（这里是 i）
    #global_step：x轴的值（这里也是 i）
writer.close() #关闭，必写

~~~

#### writer.add_scalar的用法：

~~~python
writer.add_scalar(
    标签名,    # 字符串，图表叫什么名字
    y值,       # 数字，纵坐标（要记录的值）
    x值        # 数字，横坐标（步数/轮次）
)
~~~

### 3.代码运行

#### 运行之后：

![image-20260528162701168](C:\Users\35641\AppData\Roaming\Typora\typora-user-images\image-20260528162701168.png)

会出现一个文件夹，里面有画好的图。

#### 再在终端输入：

![image-20260528162818729](C:\Users\35641\AppData\Roaming\Typora\typora-user-images\image-20260528162818729.png)

#### 点击网址：

<img src="C:\Users\35641\AppData\Roaming\Typora\typora-user-images\image-20260528162918690.png" alt="image-20260528162918690" style="zoom:50%;" />

### 4.疑问：为什么画出来的图不是y=x？

![image-20260528164632318](C:\Users\35641\AppData\Roaming\Typora\typora-user-images\image-20260528164632318.png)

Smoothed影响了，将其拉到0就可以了。

## 四、tensorboard中writer.add_image

### 1.writer.add_image 的用法：

writer.add_image可以在tensorboard中显示图片

~~~python
writer.add_image(
    图片名字,    # 字符串，给这张图起个名
    图片张量,    #可以是tensor，numpy.array
    步长,        # x轴步数，可不写
    dataformats=''  # 
)
~~~

### 2.代码示例：

~~~python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("logs")
img_pil = Image.open("Data_BeesAndAnts/train/ants/0013035.jpg")
img_array = np.array(img_pil)
writer.add_image("test", img_array,2,dataformats='HWC') #声明我的图片格式是 高度 H、宽度 W、通道 C，因为 PIL /numpy 图片都是这个顺序。
writer.close()
~~~

#### 图片的格式：

1. **H W C（人类 / 图片 / PIL/numpy 用）**

顺序：**高度 → 宽度 → 通道**

- 你打开的手机照片
- PIL 打开的图
- numpy 数组

2. **C H W（PyTorch/TensorBoard 用）**

顺序：**通道 → 高度 → 宽度**

- 这是**神经网络专用格式**
- PyTorch 默认要这个

###  3.运行后：

<img src="C:\Users\35641\AppData\Roaming\Typora\typora-user-images\image-20260528171435503.png" alt="image-20260528171435503" style="zoom:50%;" />

在tensorboard中则会出现图片
