# YOLO

YOU ONLY LOOK ONCE

之前只是简单了解过李沐老师讲解的YOLO，这次来系统学习并进行代码实战

yolo v1提出思想后   ---》yolov10   不断完善思想

yolo在进行时将物体长宽比例进行过学习，通过比例找到物品区域

一般划分为奇数个区域，

![image-20240531100036347](https://gitee.com/ai-yang-chenxu/img/raw/master/img/image-20240531100036347.png)

 思想就是将各种优良算法组合

![image-20240531100309048](C:\Users\86186\AppData\Roaming\Typora\typora-user-images\image-20240531100309048.png)

图像的增强

![image-20240531100616277](https://gitee.com/ai-yang-chenxu/img/raw/master/img/image-20240531100616277.png)





采用PAN设计思路

backbone特征提取

自下而上 自上而下 特征增强





c2f类

c2f模块经过两个卷积层对输入数据进行特征转换

cv1卷积层将输入数据通道数*2  cv卷积层将一系列的操作后的特征图通道数从 

实现多尺度融合  提升可扩展性   保持大小不变 丰富信息  特征增强





多尺度锚框







v4到v9没啥意思 加很多trick 各个地方搬

yolo 

长宽比例差不多

有了先验知识

darknet主回路 副回路

resize 会拉伸 但是加灰条不会改变

 大特征小目标

小特征大目标

