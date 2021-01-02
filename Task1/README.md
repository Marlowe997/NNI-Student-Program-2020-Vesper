# Task1 入门任务

## NNI体验文档

### 1. AutoML工具比较

#### 1.1 MLBox

![avatar](https://mlbox.readthedocs.io/en/latest/_images/logo.png)

##### 功能

- 快速进行数据读取和分布式数据预处理/清洗/格式化。
- 高度可靠的特征选择和信息泄漏检测。
- 高维空间中精确超参数优化。
- 最新的分类和回归预测模型（Deep Learning，Stacking，LightGBM……）。
- 用模型解释进行预测。

##### 特点

- 漂移识别（Drift Identification）：使训练集的分布和测试集相类似
- 实体嵌入（Entity Embedding）：一种类别特征（categorical features）编码技术
- 超参数优化

![avatar](https://pic2.zhimg.com/80/v2-3986db40e95d418fedbcbf9291074555_720w.jpg)

##### 缺点

- 仍处在活跃的开发阶段
- 不支持无监督学习
- 只能进行最基本的特征工程，大多数情况下仍需要人工干预
- 有同时移除有用变量的风险
- 并不是真正意义上的自动机器学习工具

#### 1.2  H2O

![avatar](https://i.loli.net/2021/01/02/mcCTYa7xnGoNd56.png)

##### 功能 & 特点

- 支持R和Python

- 支持最广泛使用的统计和机器学习的算法，包括DRF，GBM，XGBoost，DL等

- 具有模型解释能力

- 支持回归和分类任务，AutoMl的功能只支持有监督任务

- 自动化

- - 建立Web的交互界面，允许用户直接交互进行机器学习操作
  - 自动进行特征工程，模型验证、调整、选择和部署
  - 自动可视化

![avatar](https://img-blog.csdn.net/20170503151549180?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2lsZF9qZWVr/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

#### 1.3 TPOT

![avatar](https://raw.githubusercontent.com/EpistasisLab/tpot/master/images/tpot-logo.jpg)

​	TPOT是一个使用genetic programming算法优化机器学习piplines的Python自动机器学习工具，TPOT 通过智能地探索数千种可能的 piplines 来为数据找到最好的一个，从而自动化机器学习中最乏味的部分。一旦 TPOT 完成了搜索，它就会为用户提供Python代码，以便找到最佳的管道，这样用户就可以从那里修补管道，最后会输出最佳模型组合及其参数 ( python 文件) 和最佳得分。

##### 优点

- 分析过程较科学，在数据治理阶段采用了 PCA 主成份分析，在模型选择过程中可以使用组合方法。
- 能直接生成一个写好参数的python 文件。

##### 缺点

- 输出可参考的结果较少

### 2.NNI安装及使用



### 3.NNI使用感受



## NNI样例分析文档

