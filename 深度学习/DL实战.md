# 数据处理

#### sorted(set())-->set的意思是将其提取成随机不重复序列，用于提取较多label时使用

```python
leave_labels = sorted(set(train_data['label']))
```

#### zip 将两个长度相同的可迭代对象一一对应返回元组 dict将元组打包成字典

```python
class_to_num = dict(zip(leaves_labels, range(n_classes)))
```

#### 最后反向将num转化成类别

```python
num_to_class = {v : k for k, v in class_to_num.items()}
```

这是一次示例,是处理叶子的图片的，此时叶子图片的大小不同，以jpg文件形式存储在文件夹中

```python
import os
from torch.utils.data import DataLoader,Dataset
class MyDataSet(Dataset):
    def __init__(self,mode,transforms):
        self.transform = transforms
        self.image_base_path = r'C:\Users\86186\Desktop\classify-leaves'
        label = pd.read_csv(r'C:\Users\86186\Desktop\classify-leaves\train.csv')
        inf = label.to_numpy().tolist()
        # 将inf中第二列的所有类别转换成数字
        leaves_labels = sorted(set(labels_dataframe['label']))
        n_classes = len(leaves_labels)
        class2num = dict(zip(leaves_labels,range(n_classes)))
        for i in inf:
            i[1] = class2num[i[1]]
        # 划分数据集
        if mode == 'train':
            self.inf = inf[:20000]
        else:
            self.inf = inf[20000:]
    def __len__(self):
        return len(self.inf)
    def __getitem__(self, index):
        ds_labels = self.inf[index][1]
        ds_img = self.inf[index][0]
        ds_img = os.path.join(self.image_base_path,ds_img)
        ds_image = Image.open(ds_img).convert('RGB')
        ds_image = self.transform(ds_image)
        return ds_image,ds_labels
transorform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Resize((244,244)),torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
train_dataset = MyDataSet('train',transorform)
train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=64)
train_iter = iter(train_data_loader)
image,label = train_iter.__next__()
def image_show(img):
    img = img.numpy()
    plt.imshow(np.transpose(img,(1,2,0)))
image_show(torchvision.utils.make_grid(image))
```

dataloader使用是最后有一个参数droplast，在batchsize不是1时使用，以便所有输入都相同

