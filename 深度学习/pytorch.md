# pytorch框架

###### dir工具与help工具 十分有用

流程 首先要介绍 dataset与dataloader类

## dataset

提供一种方式去获取数据及其label值，数据集类可以由自己定义，一般进行一次任务时，需要构建新的数据集

接下来提供一次示例，为叶子分类竞赛中的一次尝试

## torchvision.datasets.ImageFolder

用于直接构造文件夹为类别，文件夹内为图片的文件结构

### dataloader

数据加载器，并且提供了一定的采样方法

为后面的网络提供不同的数据形式

告诉我们总共有多少的数据

## 一、tensors

### 张量就是矩阵 重点是可以使用GPU进行快速运算

torch.tensor以及torch生成的各种矩阵类型数据，都是张量

通过.size可以实现对其大小的获取 得到一个元组

通过torch的函数可以实现对于张量的计算 还有原地操作 如add_值得学习

### 对torch张量的转换十分简单

使用.numpy就可以将张量转化为numpy格式

所有在CPU上的张量 除了字符张量都可以在numpy之间转换

### CUDA张量

使用.cuda函数可以将张量移动到GPU上进行操作

## 二、自动求导

### 自动求导是神经网络的核心，使用autograd包

autograd.Variable是其包的核心类，其包装了张量，可以支持几乎所有张量上的操作，一旦完成前向计算，可以通过.backward方法自动计算所有梯度 可以使用.data方法访问变量中的所有张量，这个变量的梯度被计算放入.grad属性中



样本的读取 有了训练集 测试集 开始进行模型训练

## 训练需要 

迭代器（小批量读取）

```python
def data_iter(batch_size,features,labels):
	indices = list(range(len(features)))
	random.shuffle(indices)
	for i in range(0,len(features),batch_size):
		batch_indeices = torch.tensor(indices[i:min(i+batch_size,len(features))])
		yield features[batch_indices],labels[batch_indices]
```

```伪代码
#伪代码版本
def 迭代器：
	num = 批次
	list （所有数据的下标）
	打乱 list 
	循环 num次 每次越过一个批量大小
		将批量大小中的下标 存储并转换成tensor
		返回一个迭代器 返回每个批次下标对应的特征以及标签
```

使用时

```python
for X,y in data_iter(batch_size,features,labels):
```

训练需要损失函数（返回） 模型（有输入 有输出）优化器（参数输入 学习率 批量大小）

然后定义超参数 num_epochs(训练次数)

```python
def train():
	for epoch in range(num_epochs):
    	for X,y in data_iter(batch_size,features,labels):
            loss = loss(net(X,w,b),y)
            loss.sum().backward()
            sgd([w,b],lr,batch_size)
         with torch.no_grad():
            train_l = loss(net(features,w,b),labels)
```

##  简洁实现

```python
import torch.utils import data

```

```python
#迭代器
def load_array(data_arrays,batch_size,is_train=True):
	dataset = data.TensorDataset(*data_arrays)
    #在python中  *代表解包操作 将一个包含多个元素的元组，列表，集合等数据结构解压为多个独立的元素
	return data.DataLoader(dataset,batch_size,shuffle = is_train)
```

```python
#简洁模型
from torch import nn

net = nn.Sequential(nn.Linear(2,1))

#初始化模型参数
net[0]——》指的是net的第一层

net[0].weight.data.normal(0,0.01)
net[0].bias.data.fill(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(),lr = 0.03)
```

```python
#模型训练
num_epochs = 3
for epoch in range(num_epochs):
	for X,y in data_iter:
		loss = loss(net(X),y)
		trainer.zero_grad()
		l.backward()
		trainer.step()
	l = loss(net(features),labels)
```

# softmax实现

```python
#实现分类
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition
```

 经过softmax处理的值可以成为每行和为一的概率

```python
#交叉熵损失
def cross_entropy(y_hat,y):
	return -torch.log(y_hat[range(len(y_hat)),y])
```

```python
def accuracy(y_hat,y):
	if len(y_hat.shape)>1 and y_hat.shape[1]>1: """张量大于一维且列数大于一"""
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())        """"这里得到的类型是布尔类型，需要进行转换"""
accuracy(y_hat,y)/len(y)
```

```python
def evaluate_accuracy(net,data_iter):
	if isinstance(net,torch.nn.Module):
		net.eval()
	metric = Accumulator(2)
	for X,y in data(iter):
		metric.add(accuracy(net(X),y),y.numel())
	return metric[0]/metric[1]
```

```python
def train_epoch_ch3(net,train_iter,loss,updater):
	if instance(net,torch.nn.Module):
		net.train()
	metric = Accumulator(3)
	for X,y in train_iter:
		y_hat = net(X)
		l = loss(y_hat,y)
		if isinstance(updater,torch.optim.Optimizer):
			updater.zero_grad()
			l.backward()
			updater.step()
			metric.add(
				float(l)*len(y),accuracy(y_hat,y),y.size().numel()
            )
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y_hat,y),y.size().numel())
	return metric[0]/metric[2],metric[1]/metric[2]
#return loss.mean,accuracy.mea

               
		
```

简洁实现

```python
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init_weights)
```

```python
loss = nn.CrossEntropyLoss()
updater = torch.optim.SGD(net.parameters(),lr = 0.1)
num_epochs = 10
def train(net,train_iter,updater,loss,num_epochs):
    for X,y in train_iter:
        y_hat = net(X)
        l=loss(y_hat,y)
        updater.no_grad()
        l.backward()
        updater.step()
        return float(l)*len(y)/X.size().numel() 
```

## 实现多层感知机

```python
num_input,outputs,num_hiddens=784,10,256
W1 = nn.parameter(num_input,num_hiddens,requires_grad=True)
b1 = nn.parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.parameter(num_hiddens,outputs,requires_grad=True)
b2 = nn.parameter(torch.zeros(outputs,requires_grad=True))

params = [W1,b1,W2,b2]
```

实现激活函数

```python
def relu(X):
	a = torch.zeros_like(X)
	return torch.max(X,a)
```

实现模型

```python
def net(X):
	X = X.reshape((-1,num_inputs))
	H = relu(X @ W1+b1)
	return (H @ W2+b2)
loss = nn.CrossEntropyloss()
```

## 深度学习计算

```
#%%
# coding:utf-8
import torch
import numpy as np
from torch import nn

X = torch.rand(100, 20)


class MySequencial(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
            return X

net = MySequencial(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
net.forward(X)
#%%
X = torch.rand(2,4)
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU()
                         ,nn.Linear(8,4),nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block{i}",block1())
    return net

rnet = nn.Sequential(block2(),nn.Linear(4,1))
rnet(X)
rnet[0][0][0].weight.data
#%%
import torch
from torch import nn
net = nn.Sequential(nn.Linear(4,8),
                    nn.ReLU(),
                    nn.Linear(8,1))
X = torch.rand(size=(4,4))
net(X)
print(f'a{net[0].weight}')
def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
xavier_init(net[0])
net[0].weight
#%%
```

### 参数绑定

在例如CNN这样的神经网络中可能会出现共享层的情况

我们使用参数绑定方法

```python
share = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.Relu(),
					share,nn.Relu(),share,nn.Relu(),nn.Linear(8,1))
net(X)
#此时开始 对于net的第二层与第四层已经进行了绑定 更改其中一个的参数会同时更改另一个
```

## 自定义层

我们接下来自定义一个层

```python
import torch
from torch import nn 
class CenteredLayer(nn.Mudule):
	def __init__(self):
		super().__init__
	def forward(self,X):
		X-=X.mean()
net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y = net(torch.rand(4,8))
Y.mean()
```

我们以上做的就是定义了一个层 将所有值都减去均值 从而使得均值为0 

本次的暑期组队学习为我们提供了很多的教程与讲解，让我学到了很多机器学习的专业知识与技能，实现了从数据竞赛小白到初学者的蜕变，并且能够在比赛中获得不错的成绩，同时也让我对自己的专业有了更为清晰的认知，有了更明确的学习方向。

```python
class MyLinear(nn.Module):
	def __init__(self,in_units,units):
		super().__init__()
		self.weight = nn.Parameter(torch.randn(in_units,units))
		self.bias = nn.Parameter(torch.randn(units,))
	def forward(self,X):
		linear = torch.matmul(X,self.weight.data) + self.bias.data
		return F.relu(linear)
net = nn.Sequential(MyLinear(64,8),MyLinear(8,1))
net(torch.rand(2,64))
```

这里都是自我定义出的，相当于了一个Linear层，所以在forward层需要前向计算



## 参数保存

```python
import torch
X = torch.arange(4)
torch.save(X,'X-file')
A = torch.load('X-file')
A
```

```python
Y =torch.zeros(4)
torch.save(Y,'Y-file')
Y
```

```python
torch.save([X,Y],'X-Y-file')
x,y = torch.load('X-Y-file')
```

```python
my_dict = {'x1':X,'y1':'123'}
torch.save(my_dict,'dict-file')
torch.load('dict-file')
```

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.output = nn.Linear(256,10)
    def forward(self, x1):
        return self.output(nn.functional.relu(self.hidden(x1)))
net = MLP()
X = torch.rand(size=(2,20))
Y = net(X)
 #state_dict是我们的参数字典我们存储就行
torch.save(net.state_dict(),'MLP.params')
```

```python
clone = MLP()
clone.load_state_dict(torch.load('MLP.params'))
clone.eval()#eval是评估模式 返回值是self本身
```

```python
Y_clone = clone(X)
Y_clone == Y
```

# Q A:

##### 当转换成伪变量时内存爆炸 可以储存为稀疏矩阵，是常用的做法

##### 针对类中的forward函数，是继承的Module类的一些方法，使用了call魔术方法，很神奇，是对着paper上的数学公式实现的

# 卷积

```python
import torch
from torch import nn
def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros(X.shape[0]-h+1,X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w] * K).sum()
    return Y
```

```python
X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
K = torch.tensor([[0.0,1.0],[2.0,3.0]])
corr2d(X,K)
```

```python
X = torch.ones(6,8)
X[:,2:6] = 0
K = torch.tensor([[1.0,-1.0]])
Y = corr2d(X,K)
corr2d(X.t(),K.t())
```

```
conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias = False)
X = X.reshape((1,1,6,8))
Y = Y.reshape((1,1,6,7))
for i in range(100):
    Y_hat = conv2d(X)
    l = (Y_hat - Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= 3e-2 *conv2d.weight.grad
    if (i + 1 ) % 2 == 0:
        print(f'batch{i+1},loss{l.sum():3f}')
```

```python
conv2d.weight.data.reshape((1,2))
```
