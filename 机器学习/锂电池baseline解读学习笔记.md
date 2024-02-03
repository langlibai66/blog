# 学习笔记

## 学习解读baseline与lightgbm学习

### 导入所需的库有
pandas—>用作数据处理

lightgbm—>模型

sklearn.metrics.mean_absolute_error—>mae计算函数

sklearn.model_selection.train_test_split—>数据集拆分工具 

tqdm—>进度条工具

### 数据准备

```Python
train_dataset = pd.read_csv("./data/train.csv") # 原始训练数据。
test_dataset = pd.read_csv("./data/test.csv") # 原始测试数据
submit = pd.DataFrame() # 定义提交的最终数据。
submit["序号"] = test_dataset["序号"] # 对齐测试数据的序号。
MAE_scores = dict() # 定义评分项。
```

### 设定 LightGBM 训练参

```python
lgb_params = {
    'boosting_type': 'gbdt',    #梯度提升方法
    'objective': 'regression',  #优化目标：回归
    'metric': 'mae',			#评估指标
    'min_child_weight': 5,		#权重最小和
    'num_leaves': 2 ** 5,		#树子节点数
    'lambda_l2': 10,			#正则化权重
    'feature_fraction': 0.8,	#特征随机选择比例
    'bagging_fraction': 0.8,	#数据随机选择比例
    'bagging_freq': 4,			#数据随机选择频率
    'learning_rate': 0.1,		#学习率
    'seed': 2020,				#随机种子
    'nthread': 28,				#并行线程数
    'verbose': -1,				#日志输出
}
```

##### 参数选择

##### 我采用了此套参数将分数训练至6.23961

### 特征提取

#### 将序号drop 将时间转化为可训练类型 将原时间数据drop

#### 在提取前将数据复制一份

### 从所有待预测特征中依次取出标签进行训练与预测。  

### 训练模型，参数依次为：导入模型设定参数、导入训练集、设定模型迭代次数、导入验证集、禁止输出日志

以及以下是

# lightgbm学习笔记

XGBoost—>梯度提升树框架的一个里程碑

CatBoost—>对离散特征数据进行了优化

LightGBM—>通过梯度采样与直方图算法支持并行化

## XGBoost

Boosting往往串行生成学习器  Bagging系列算法往往并行生成

串行并行结合 在数据处理过程中使用并行运算

大规模端到端模型 有效处理稀疏数据与缺失数据

提升模型性能：从基学习器入手

​						   从误差优化入手

数学推导慢慢补

寻找最优结构 通过启发式搜索 贪心搜索 

总之就是将残差值降低以改进模型整体性能

成为机器学习里程碑，优化了梯度，提高了准确度，能自主处理数据缺失，削弱预处理的困难，能自主感知数据稀疏，减少人工降维工作量。





   