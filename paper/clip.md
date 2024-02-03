# CLIP论文精读

## CLIP是什么

一个强大的无监督训练模型

通过NLP来的监督信号得到迁移学习

进行图片与文字的配对实现监督的信号，解决了需要打标签进行训练的限制，增强了模型的泛化能力

## CLIP结构

CLIP的结构包含两个模型**Text Encoder**和**Image Encoder**，Text Encoder用于提取文本特征，Image Encoder用来提取图像特征

### CLIP训练

![img](https://pic2.zhimg.com/80/v2-b86361b47d4db80258439b8ad33bdf8d_720w.webp)

CLIP的训练数据是**图像-文本对**，如图上方是对小狗的描述，而下方是这张图片，通过对文本的特征提取与对图像的特征提取进行对比学习，对于N个图像文字对，预测出$N^2$个相似度，这里的相似度直接结算文本特征和图像特征的余弦相似性，实际上真实对应的相似对是位于对角线上的元素，我们的目的就是最大化对角线上的元素而减小非对角线上的元素 

### 实现zero-shot分类

![img](https://pic4.zhimg.com/80/v2-acd4b008007ca7de78bdab1c9042bbcb_720w.webp)

首先先将分类标签扩充成句子后输入到 TextEncoder中，而进行分类时的标签并不需要是训时存在的标签 ，你完全可以新加一个背带裤的标签进行分类，训练与推理时都没有标签的限制，属实是将视觉与文字的语义相关性真正学习到了。

 使用clip可以辅助实现风格迁移，AI换脸换衣，图像检测 分割，视频检索

## 论文部分

采用有限制性的监督信号会限制模型的泛化性这一点毋庸置疑 ，要识别新的物体类别时候就有了困难

所以CLIP的想法就是由语言生成监督信号 

经过测试，CLIP在ImageNet上可以跟专门为了ImageNet训练出来的resnet50打成平手 达到了非常好的效果

并且可以随着两个模型性能继续增长后可以达到不断的进步

从文本出来的弱监督信号不实用，是因为数据量不足够与算力消耗大

方法上实际上都差不多，但是数据量规模是其想成长的必要因素

像VirTex，ICMLM，ConVIRT这些工作都进行过类似的尝试，但是都是只训练了几天的规模，并不足以达到很好的效果

于是openAI团队为了证明这一点 收集了超级大规模的数据想要达到比较好的效果

再加上大模型的加持，可以达到非常不错的效果，这就是**CLIP**（**Contrastive Language-Image Pre-training**）

对于模型选择，作者团队也尝试了多种的尝试，发现CLIP的效果跟模型规模是有正相关的

最终得到的效果是，CLIP在30多个数据集上基本都能与精心设计的模型打成平手甚至胜利，并且会有更好的泛化性

使用CLIP的好处有很多，其中之一就是CLIP不需要再对数据进行标注，只需要获得文本—图像对就可以，像在社交平台上获得的图片跟他发布时的TAG就是一个很好的途径，这种数据往往比标注的数据更容易获取，另外，通过文本—图像对的这种数据集训练，使得其拥有了多模态的效果 



在预训练过程中，作者团队采用了对比学习的方法，之所以使用这样的方法而不是用GPT就是因为语言的多样性导致对应关系有很多（例如一张图片可以从多个角度描述），所以我们只需要让图片与文本配对即可，通过这样就能达到很高的效率

## 代码的实现

在实现方面，通过论文所给伪代码

```python
# image_encoder 	- ResNet or Vision Transformer
# text_encoder 		- CBOW or Text Transformer
# I[n, h, w, c] 	- minibatch of aligned images
# T[n, l] 			- minibatch of aligned texts
# W_i[d_i, d_e] 	- learned proj of image to embed
# W_t[d_t, d_e] 	- learned proj of text to embed
# t 				- learned temperature parameter

# 分别提取图像特征和文本特征
I_f = image_encoder(I) 	#[n, d_i]
T_f = text_encoder(T) 	#[n, d_t]
# 在得到特征时一般会尝试归一化 在归一化前，还涉及到了投射层，即np.dot(I_f, W_i)，主要用来学习如何从单模态投射到多模态 
# 对两个特征进行线性投射，得到相同维度的特征，并进行l2归一化
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# 计算缩放的余弦相似度：[n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# 对称的对比学习损失：等价于N个类别的cross_entropy_loss
labels = np.arange(n) 	# 对角线元素的labels
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

步骤解释

1. 提取图像特征和文本特征：
   - 使用预训练的 image_encoder 和 text_encoder 分别提取图像 I 和文本 T 的特征，得到形状为 [n, d_i] 和 [n, d_t] 的特征向量。
2. 线性投射和归一化：
   - 对两个特征进行线性投射，分别用矩阵 W_i 和 W_t，得到相同维度的特征向量 I_e 和 T_e。
   - 对投射后的特征进行 l2 归一化，保证它们具有单位长度。
3. 计算缩放的余弦相似度：
   - 通过计算余弦相似度矩阵，得到形状为 [n, n] 的 logits 矩阵。这一步涉及将图像特征和文本特征进行相似度计算，并使用温度参数 t 进行缩放。
4. 对称的对比学习损失：
   - 创建标签 labels，其中包含了元素从 0 到 n-1。
   - 计算两个方向上的交叉熵损失：loss_i 是以图像为查询，文本为正样本的损失；loss_t 是以文本为查询，图像为正样本的损失。
   - 最终的损失是两个方向上损失的平均值，即 (loss_i + loss_t) / 2。

这个损失函数的设计旨在通过最大化正样本之间的相似度、最小化负样本之间的相似度，来学习图像和文本之间的语义对应关系。这种对称性的设计可以帮助提升模型的泛化能力，使得图像和文本之间的表示更加一致和可靠。
