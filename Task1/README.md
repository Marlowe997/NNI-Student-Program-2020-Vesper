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

```
 $ conda create -n nni python=3.8
 $ conda activate nni
 $ conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
 $ pip insatll nni
 $ git clone -b v1.9 https://github.com/Microsoft/nni.git
 $ nnictl create --config nni\examples\trials\mnist-pytorch\config_windows.yml
```

创造一个环境并启用

## NNI样例分析文档

### 1 mnist-pytorch样例测试流程

#### 1.1 基本信息

- 测试平台：windows 10
- 测试环境：Anaconda 3
- conda版本：4.9.2
- nni版本：1.8
- torch版本：1.7.0

#### 1.2 基本运行代码

##### 1.2.1 配置文件：mnist-pytorch\config_windows.yml

```yml
authorName: default
experimentName: example_mnist_pytorch
trialConcurrency: 1    #同时运行trial的个数
maxExecDuration: 2h    #整个调参过程最大时长
maxTrialNum: 10        #NNI最大Trial任务数
#choice: local, remote, pai  
trainingServicePlatform: local #指定Experiment 运行的平台为local
searchSpacePath: search_space.json #搜索空间
#reset nni_experiment dir
logDir: E:/AI/nniGit_experiments  #日志路径
logLevel: info
#choice: true, false
useAnnotation: false  #不删除searchSpacePath字段
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE #内置调参算法为TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python mnist.py
  codeDir: .  #Trial文件的目录
  gpuNum: 0   #每个trial运行时gpu个数
```

##### 1.2.2 搜索空间： search_space.json

```json
{
	#定义了一次训练时所选取的样本数
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
	#定义了隐藏层尺寸
    "hidden_size":{"_type":"choice","_value":[128, 256, 512, 1024]},
	#定义学习率
    "lr":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]},
	#定义动量
    "momentum":{"_type":"uniform","_value":[0, 1]}
}
```

以上，可得训练次数设置为10；最长运行时间2h；tuner为TPE。

#### 1.3 运行结果

Overview

![image.png](https://i.loli.net/2021/01/03/QBWaoDud6t9myCf.png)

Trial Detail & Default Metric

![image.png](https://i.loli.net/2021/01/03/jrzvgdIMlihepXL.png)

![image.png](https://i.loli.net/2021/01/03/568tMryjvqD27IA.png)

Hyper Parameter

![image.png](https://i.loli.net/2021/01/03/w7qXVpmfJsHMaex.png)

Trial Duration

![image.png](https://i.loli.net/2021/01/03/UfOFGSy56u7cwtQ.png)

Intermediate result
![image.png](https://i.loli.net/2021/01/03/nFaQB9ACSRgsobk.png)

#### 1.4 结果分析

- 十次测试中，有两次准确率极低，结合Hyper Parameter图像可知，主要因为学习率learnrate太大，一般情况下，当学习率过小时，收敛过程会变得十分缓慢，会导致代码运行时间过长的现象，而当学习率过大时，梯度可能会在最小值附件来回震荡，甚至可能无法收敛，本次调参过程中，两次准确率较低都是因为此原因。因此，在实际神经网络学习过程中，一般学习率不能过大也不宜过小。
- batch_size的大小直接决定了训练所需时长，两者负相关，但与精度正相关，即batch_size越大，在同等精度时，需要的epoch越大，训练时长也越长。
#### 1.5 使用体验

- NNI有着良好的图形界面，安装方便等特点，能够从图形界面较为直观地看出训练结果

---

#### 使用Windows Terminal 时遇到的问题

在使用windows terminal 时，nni运行10s从waiting转到fail状态，无错误log，经检查，原来是anaconda未激活环境，解决方法,在windows terminal中`` code $profile`` 增加以下代码

```powershell
function Start-Anaconda() {
    C:\ProgramData\Anaconda3\shell\condabin\conda-hook.ps1
    conda activate 'C:\ProgramData\Anaconda3'
}
```

每次启动base 环境时使用``Start-Anaconda``即可