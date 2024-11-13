```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```


```python
import torchvision.transforms as transform
from torchvision import transforms

class dataset(Dataset):
    def __init__(self,df,transform = None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:,0].values
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image  = self.transform(image)
        else:
            image = torch.tensor(image/255.,dtype=torch.float)
        label = torch.tensor(label,dtype=torch.long)
        return image,label
image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),
     # 这一步取决于后续的数据读取方式，如果使用内置数据集读取方式则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])
train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')
train_data = dataset(train_df,data_transform)
test_data = dataset(test_df,data_transform)
train_data_loader = DataLoader(train_data,shuffle=True,batch_size=4)
test_data_loader = DataLoader(test_data,shuffle=False,batch_size=4)
```


```python
import matplotlib.pyplot as plt
image, label = next(iter(train_data_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")
```

    torch.Size([4, 1, 28, 28]) torch.Size([4])
    




    <matplotlib.image.AxesImage at 0x22f391c3d90>




    
![png](test_files/test_2_2.png)
    



```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1,3,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(),
            nn.Conv2d(3,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.FC1 = nn.Linear(32*5*5,10)
    def forward(self,x):
        x = self.block1(x)
        x = self.avgpool(x)
        x = x.view(-1,800)
        x = self.FC1(x)
        return x
model = Net()
model = model.cuda()
```


```python
loss_F = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
```


```python
def train(epoch):
    model.train()
    train_loss = 0
    for data,label in train_data_loader:
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        result = model(data)
        loss = loss_F(result,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss / len(train_data_loader.dataset)
    print('Epoch:{} \tTraining_loss:{:.6f}'.format(epoch,train_loss))

def val(epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data,label in test_data_loader:
            data,label = data.cuda(),label.cuda()
            output = model(data)
            preds = torch.argmax(output,1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = loss_F(output,label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_data_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
for epoch in range(1,6):
    train(epoch)
    val(epoch)

```

    Epoch:1 	Training_loss:0.604881
    Epoch: 1 	Validation Loss: 0.659659, Accuracy: 0.788900
    Epoch:2 	Training_loss:0.572515
    Epoch: 2 	Validation Loss: 0.658802, Accuracy: 0.792300
    Epoch:3 	Training_loss:0.554421
    Epoch: 3 	Validation Loss: 0.675458, Accuracy: 0.772900
    Epoch:4 	Training_loss:0.539022
    Epoch: 4 	Validation Loss: 0.629025, Accuracy: 0.795200
    Epoch:5 	Training_loss:0.529567
    Epoch: 5 	Validation Loss: 0.614800, Accuracy: 0.798700
    


```python
save_path = './FashionModel.pkl'
torch.save(model,save_path)
```


```python

```
