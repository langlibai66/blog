# Sora训练营打卡

本次组队学习对于Sora的一些基础内容进行了讲解，由于学生党开学，再加上有些比赛，近期事情比较多，只能给出直播课时的随堂笔记加以补全，助教老师评选时手下留情，简单看看即可

## 简介

Sora是openAI最新发布的视频生成模型，所发布的内容较之前的视频生成模型在各方面都有了很大的提高，不仅在视频生成上视频的长度得到了增长，达到了60s，还拥有了很高的真实性，相比于之前容易变形的生成内容，可以说是有了非常巨大的进步，甚至还能够生成多角度多机位视频，支持任意比例视频，将两个视频丝滑的融合在一起，这是之前的视频生成模型都到达不了的高度

> 杨老师PPT里所讲解能力如下
>
> TeText-to-video: 文生视频
>
> Image-to-video: 图生视频
>
> Video-to-video: 改变源视频风格or场景
>
> Extending video in time: 视频拓展(前后双向)
>
> Create seamless loops: Tiled videos that seem like they never endImage generation: 图片生成 (size最高达到 2048 x 2048)
>
> Generate video in any foformat: From 1920 x 1080 to 1080 x 1920 视频输出比例自定义
>
> Simulate virtual worlds: 链接虚拟世界，游戏视频场景生成
>
> Create a video: 长达60s的视频并保持人物、场景一致性

虽然目前Sora在物理引擎方面还有这不小的欠缺，生成视频不符合物理的客观规律，通过官方所说：如，一个人可能会咬一口饼干，但之后，饼干可能没有咬痕。

虽然Sora还在内测过程中，但是openAI已经发布了官方文档，文档中为大家提供了很多的技术实现信息

## Sora训练流程

### 核心技术陈列

Sora使用的一个很关键的步骤就是将原始视频数据切分为 Patches通过 VAE 编码器压缩成低维空间表示

所谓patch，就是将视频数据压缩成为一维的向量表示，这就使得不同类型的数据也能够统一参与训练

einops是一个用于操作张量的库,它的出现可以替代我们平时使用的reshape、view、transpose和permute等操作，通过这个库，我们就能实现patch化

输出在时间和空间上都经过压缩的潜在表示。Sora在这个压缩的潜空间中接受训练并随后生成视频。此外还训练了一个相应的解码器模型，该模型将生成的潜在对象映射回像素空间

Sora是一个在不同时长、分辨率和宽高比的视频及图像上训练而成的扩散模型，同时采用了Transfoformer架构

与 GPT 模型类似，Sora 使用 transformer 架构，释放出卓越的扩展性能。

它使用了 DALL·E 3，涉及为视觉训练数据生成高度描述性的标题。因此，该模型能够更忠实地遵循生成视频中用户的文本说明。

他所进行的生成过程就是一个人工加噪声再进行去噪的过程，使用DDPM扩散模型

VIT技术就是让图像技术得以使用transformer技术，将图像分为多个patch后直接应用于图像，图像被划分为多个 patch后，将二维 patch 转换为一维向量作为 Transformer 的输入

### 时空数据理解

针对时空数据的理解，杨老师给了我们一种叫做摊大饼的方法

摊大饼法：从输入视频剪辑中均匀采样 n_t 个帧，使用与ViT相同的方法独立地嵌入每个2D帧(embed each 2D frameindependently using the same method as ViT)，并将所有这些token连接在一起

### DIT

结合 Diffusion Model 和 和 Transformer，通过 Scale up Model 提 提升图像生成质量图像的scaling技术运用到视频场景非常直观，可以确定是  SORA  的技术之一

DiT 利用 transformer 结构探索新的扩散模型，成功用 transformer 替换 U-Net 主干

### 保持视频信息

> OpenAI 使用类似 DALLE3 的Cationining 技术训练了自己的 Video Captioner用以给视频生成详尽的文本描述
>
> 模型层不通过多个 Stage 方式来进行视频预测而是整体预测视频的 Latent在训练过程中引入 Auto Regressive的task帮助模型更好地学习视频特征和帧间关系



但是确定这些信息，仍然不代表我们就能够复现出一个Sora，这些只是独立的技术，我们需要的是将这些技术通过合理手段拼接起来