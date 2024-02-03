# autoML学习

使用工具

## auto_ml

####  它主要将机器学习中所有耗时过程自动化，如数据预处理、最佳算法选择、超参数调整等，这样可节约大量时间在建立机器学习模型过程中。

进行自动机器学习

使用的库为pycaret

## pycaret——》开源机器学习库

### 不好用

从数据准备到模型部署 一行代码实现

可以帮助执行端到端机器学习试验 无论是计算缺失值 编码分类数据 实施特征工程 超参数调整还是构建集成模型 都非常方便

使用前新建虚拟环境：

scikit-learn==0.23.2

pycaret跟auto-ts有冲突

根据要解决的问题类型，首先需要导入模块。在 PyCaret 的第一个版本中，有 6 个不同的模块可用 ---> **回归、分类、聚类、自然语言处理 (NLP)、异常检测和关联挖掘规则**。

我们这次要预测新增用户，所以是一个分类问题，我们引入分类模块

```python
# import the classification module 
from pycaret import classification
# setup the environment 
classification_setup = classification.setup(
   data= data_classification, target='Personal Loan')
```



设置更多自定义参数

```python
data_amend = exp_mclf101 = setup(
    data= data_classification,
    target='Personal Loan', 
    train_size = 0.80,
    ignore_features = ["session_id",...],
    numeric_features =["Age",...],
    combine_rare_levels= False,
    rare_level_threshold=0.1,
    categorical_imputation = 'mode',
    imputation_type ='simple', 
    feature_interaction = True, 
    feature_ratio= True, 
    interaction_threshold=0.01,
    session_id=123,
    fold_shuffle=True, 
    use_gpu=True,  
    fix_imbalance=True,
    remove_outliers=False,normalize = True,
    transformation = False, 
    transformation_method='quantile',
    feature_selection= True,
    feature_selection_threshold = 0.8, 
    feature_selection_method='boruta',
    remove_multicollinearity = True,
    multicollinearity_threshold=0.8
    normalize_method = 'robust')
```

在我们使用过程中只需要进行调用函数，函数值接受一个参数，也就是模型缩写

这个表格包含了模型缩写字符串

![图片](https://mmbiz.qpic.cn/mmbiz_png/eWEPx9ZqUHssmZmsV4OeYUDjCUd4l2oE2rGtmDre8Q7QR10cNuBwkiaEybtoTsRlSFYcohSOszYticZK04N9iczsQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)

最后，我们将对陌生数据进行预测。为此，我们只需要传递将用于预测的数据集的模型。注意的是，确保它与之前设置环境时提供的格式相同。PyCaret 构建了所有步骤的管道，并将预测数据传递到管道中并输出结果。



## 通过拜读群内大佬的数据处理过程 得到以下技巧

分组聚合 将x1到x8的数据进行分组聚合 并计算每个分组对于target的均值

###### 猜想 有些特征数值比较大 是否使用正则化

进行时间序列上的处理 将其转化成月 日 小时 分钟 以及是否为周末 一年中的第几周

时间特征实际上并不好用

依然使用决策树进行训练

这样的训练方法使得分数进行了一定的上升，但依旧存在召回率低的问题

 
