# 数据集AV16.3

数据集使用两个 8 麦克风均匀圆形阵列（16 kHz 采样频率）和三个数码相机（每秒 25 帧）进行录音，因此得名“AV16.3”。

长8.2米

宽3.6米

高2.4米

# 主文件stGCF

## au_observe = getGCF()

getGCF()来自文件GCF_extract_stGCF.py

# 文件GCF_extract_stGCF

## GCF类

### GCFextract函数 

输入数据DATA img GCC fa cam_number以生成GCFmap

首先生成2d 与 3d样本点

选取1/9的个数的点作为采样点

#### 2d采样点

(11520, 2)

11520是采样后的点的个数 2是指x y两个维度

#### 3d采样点



这个用来定计算GCF的坐标





(9, 11520, 3)

9是mapnum，是深度，11520是采样点数，3是x y z三个维度

#### tau3d

(9, 120, 11520)

这个是通过位置算出来的，9依然是深度 11520依然是个数

120是指 16*15/2 即麦克风对的不重复组合

### cal_rGCFmax函数

m3是指查看帧数





使用此作为一个维度，进行点云构建

rGCF 			shape(15, 9, 11520)





rGCFmax 	shape(15)






样本点只是坐标点

```python
for g in range(DATA.cfgGCF.map_num):  # 对于每个GCF map的数目
            tau3dlist = np.zeros(shape=(int(pairNum), sample2d.shape[0]))  # 初始化tau3d列表
            t = 0  # 初始化麦克风对索引
            for mici in range(len(DATA.audio)):  # 对于每个麦克风
                di = np.sqrt(np.sum(np.asarray(DATA.micPos[mici] - sample3d[g]) ** 2, axis=1))  # 计算麦克风到样本点的距离
                for micj in range(mici + 1, len(DATA.audio)):  # 对于每个麦克风对
                    dj = np.sqrt(np.sum(np.asarray(DATA.micPos[micj] - sample3d[g]) ** 2, axis=1))  # 计算另一个麦克风到样本点的距离
                    tauijk = (di - dj) / DATA.cfgGCF.c  # 计算时间延迟差
                    taun = np.transpose(tauijk * DATA.cfgGCF.fs)  # 转换为时间样本数
                    taun = np.rint(taun * interp + max_shift)  # 取整并添加最大时间偏移
                    tau3dlist[t, :] = taun  # 将结果添加到tau3d列表
                    t = t + 1  # 更新麦克风对索引
            tau3d[g, :] = tau3dlist  # 将tau3d列表添加到tau3d
```

通过这段代码，将tau计算出来

然后通过时延将GCF输出