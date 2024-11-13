# Torch进阶

## 损失函数

函数方式定义的损失函数易于定义，直接写个函数就ok

不过我们也可以以**类**定义

### 以类方式定义损失函数

虽然以函数定义的方式很简单，但是以类方式定义更加常用

以类定义损失函数时，我们继承自nn.Module类，将其作为神经网络的一层看待，也可以利用自动求导

例如我们要实现

Dice Loss(分割领域常见的损失函数)
$DSC = \frac{2|X \bigcap Y|}{|X| + ||Y}$

我们的实现代码

```python
class DiceLoss(nn.Module):
	def __init__(self,weight):
		super(DiceLoss,self).__init__()
		
	def forward(self,inputs,targets,smooth=1):
		inputs = F.sigmoid(inputs)
		inputs = input.view(-1)
		targets = targets.view(-1)
		intersection = (inputs*targets).sum()
		dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
		return 1 - dice
```



## 动态调整学习率

学习率的大小会影响模型调优，我们使用scheduler方法来实现动态调整策略

pytorch中的`lr_scheduler`已经提供给了我们动态调整的方法

我们可以使用`help(torch.optim.lr_scheduler)`来查看他们的使用方法

使用示例如下

```python
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 需要在优化器参数更新之后再动态调整学习率
# scheduler的优化是在每一轮后面进行的
scheduler1.step() 
...
schedulern.step()
```

我们可以看到scheduler是在每一轮训练结束后使用的，需要放在optimizer后面进行使用

## 模型微调

我们在使用参数量比较大的网络时，不能将模型从头到尾再训练一遍

例如我们在进行特定的图像分类时可以重写分类层，前面学习到的“知识”不变，所以我们要进行模型微调

### pretrained参数

通过`True`或者`False`来决定是否使用预训练好的权重，在默认状态下`pretrained = False`，意味着我们不使用预训练得到的权重，当`pretrained = True`，意味着我们将使用在一些数据集上预训练得到的权重。

### 训练特定层

在我们只想训练特定层时，经常涉及到部分层的冻结，这就涉及到参数的属性 `require_grad `，当这个值取False时，该参数不再更新

所以我们指定函数进行层的冻结

```python
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```

## timm

使用timm.list_model进行模型查看 使用timm.create_model进行模型创建

timm模型是torch.model的子类

所以我们进行模型保存与提取时使用torch的方法即可

##  半精度训练

半精度训练能够减少显存的占用，使得同时加载更多的数据进行计算

首先我们需要引入

`from torch.cuda.amp import autocast`

 模型定义中需要使用修饰器

```python
@autocast()   
def forward(self, x):
    ...
    return x
```

在训练过程中，只需在将数据输入模型及其之后的部分放入“with autocast():“即可：

```python
 for x in train_loader:
	x = x.cuda()
	with autocast():
            output = model(x)
```

半精度训练主要适用于数据本身的size比较大（比如说3D图像、视频等）

## 数据增强

我们遇到过拟合问题的时候加入正则项或者减少模型参数 但是最简单的避免过拟合的方法就是增加数据

所以数据增强技术可以提高训练数据集的大小与质量 以便于使用他们构建更好的深度学习模型

使用imgaug进行数据增强

相比于`torchvision.transforms`，它提供了更多的数据增强方法，因此在各种竞赛中，人们广泛使用`imgaug`来对数据进行增强操作。

imgaug是一个图像增强库，并未提供IO操作，建议使用imageio进行读入，如果是OpenCV进行读入时，需要将BGR图像手动转化成RGB图像 使用PIL.Image进行读取时，因为读取没有shape属性，需要手动将img转换为np.array的形式再进行处理，官方的例程中野兽使用的imageio进行读取

### 单张图片处理

```python
from imgaug import augmenters as iaa

# 设置随机数种子
ia.seed(4)

# 实例化方法
rotate = iaa.Affine(rotate=(-4,45))
img_aug = rotate(image=img)
ia.imshow(img_aug)
```

如图就是从augmenters中使用Affine方法进行的操作，将图片随机旋转-4到45度，以便进行图片的增强

有时候我们会对一张图片做多种处理

我们使用augmenter.Sequential进行数据增强的pipline

 例如

```python
aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-25,25)),
    iaa.AdditiveGaussianNoise(scale=(10,60)),
    iaa.Crop(percent=(0,0.2))
])
```

### 对批次图片进行处理

实际使用中我们需要进行批次图片的处理

操作如下

```python
images = [img,img,img,img,]
images_aug = rotate(images=images)
ia.imshow(np.hstack(images_aug))
```

使用images参数，输入图像列表，就能达批次同效果

我们甚至可以使用imgaug进行分部分处理

使用iaa.sometimes进行比例划分，使用不同的处理方法

## 可视化

网络结构复杂的同时，我们需要一个可视化的工具来方便我们确定输入输出、模型参数

使用torchinfo可以做到可视化

使用方法非常简单，直接使用torchinfo.summary()就行了，必须的参数是model 与 input_size

但你使用的是colab或者jupyter notebook时，想要实现该方法，`summary()`一定是该单元（即notebook中的cell）的返回值，否则我们就需要使用`print(summary(...))`来可视化。

### TensorBoard

使用TensorBoard就相当于加入一个记录员，记录我们指定的数据，包括每一层的权重，训练loss等，最后使用网页形式进行可视化

```python
from tensorboardX import SummaryWriter

writer = SummaryWriter('./runs')
```

我们首先制定了writer作为记录员

pytorch中自带的TensorBoard通过以下方法引入

```python
from torch.utils.tensorboard import SummaryWriter
```

我们使用像summary中同样的思路，给定输入数据，前向传播得到模型结构，使用TensorBoard进行可视化

```python
writer.add_graph(model, input_to_model = torch.rand(1, 3, 224, 224))
writer.close()
```

对于处理图像，我们也可以在TensorBoard中进行可视化展示

对于单张图片，使用add_image

对于多张图片，使用add_images

有时需要将多张图片拼接成一张图片后用writer.add_image进行表示

```python
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose(
    [transforms.ToTensor()])
transform_test = transforms.Compose(
    [transforms.ToTensor()])

train_data = datasets.CIFAR10(".", train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(".", train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

images, labels = next(iter(train_loader))
 
# 仅查看一张图片
writer = SummaryWriter('./pytorch_tb')
writer.add_image('images[0]', images[0])
writer.close()
 
# 将多张图片拼接成一张图片，中间用黑色网格分割
# create grid of images
writer = SummaryWriter('./pytorch_tb')
img_grid = torchvision.utils.make_grid(images)
writer.add_image('image_grid', img_grid)
writer.close()
 
# 将多张图片直接写入
writer = SummaryWriter('./pytorch_tb')
writer.add_images("images",images,global_step = 0)
writer.close()
```

## torch生态

### torchvision

#### torchvision.datasets

其中包含了我们在计算机视觉领域常见的数据集

#### torchvision.transforms 

我们知道，数据集中的数据格式或者大小不一样，需要进行归一化与大小缩放等操作，都是常用的数据预处理方法，

例如

```python
from torchvision import transforms
data_transform = transforms.Compose([
    transforms.ToPILImage(),   # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])
```

#### torchvision.models

为了提高训练效率，减少不必要的重复劳动，pytorch提供给我们很多的预训练模型供我们使用

#### torchvision.io

里面提供了视频、图片和文件的IO操作等功能，包括读取、写入、编解码处理等操作

#### torchvision.ops

torchvision.ops 为我们提供了许多计算机视觉的特定操作，包括但不仅限于NMS，RoIAlign（MASK R-CNN中应用的一种方法），RoIPool（Fast R-CNN中用到的一种方法）。在合适的时间使用可以大大降低我们的工作量，避免重复的造轮子，想看更多的函数介绍可以点击[这里](https://pytorch.org/vision/stable/ops.html)进行细致查看。

#### torchvision.utils

提供了很多可视化的方法，帮助我们将若干张图片拼接在一起，可视化检测与分割的效果

### pytorchvideo

视频处理 用的时候学

#### torchtext

文本处理

#### torchaudio

音频处理
