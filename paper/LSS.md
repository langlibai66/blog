# BEV开山鼻祖 LSS

## 前置知识（2d to 3d）

内外参为这篇文章前提条件

就是相机模型

利用各种公式将坐标各种变换

## 核心思想

### Lift

将图片上每个点进行深度的估计，将特征图投影到深度上

一般图像特征提取 输出就是x y channel 

但是现在再加一个depth

 因为只是做个BEV 精度不用太高

### Splat

**拍扁**

BEV上不需要整个三维

将特征进行压缩即可，将同一个X Y 的特征累加到一起

压扁，就得到了BEV特征

现在的问题就是，深度是估计出来的，不准的话，特征投影是不正的，会投影到
