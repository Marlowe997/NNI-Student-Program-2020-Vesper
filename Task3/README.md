# Task3

## Task 3.1 NNI对于特征工程的应用

### 3.1.1 特征工程

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。

特征工程本质是一项工程活动，目的是最大限度地从原始数据中提取特征以供算法和模型使用。通过总结和归纳，人们认为特征工程包括以下方面：

![avatar](https://pic1.zhimg.com/80/20e4522e6104ad71fc543cc21f402b36_720w.jpg?source=1940ef5c)

### 3.1.2 自动特征工程

特征工程是应用经典机器学习算法的前置步骤，通过特征工程，能让机器学习过程更快得到较好的结果。而NNI 的超参调优功能，可直接应用于特征增强、自动特征选择等特征工程中的各个子领域，同时NNI 还内置了基于梯度和决策树的自动特征选择算法，并提供了扩展其它算法的接口。

正如IBM的研究者所指出的，自动特征工程能够帮助数据科学家减少在探索数据上所花费的时间，并让他们能够在短时间内尝试更多的新想法。从另一个角度来看，自动特征工程让对数据科学不熟悉的非专业人士在更短的时间内以更少的精力来发掘数据的价值。

### 3.1.3 NNI自动工程示例

#### NNI自动工程工具

**trial code**

```python
import nni

if __name__ == '__main__':
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id'

    # read original data from csv file
    df = pd.read_csv(file_name)

    # get parameters from tuner
+   RECEIVED_FEATURE_CANDIDATES = nni.get_next_parameter()

+    if 'sample_feature' in RECEIVED_FEATURE_CANDIDATES.keys():
+        sample_col = RECEIVED_FEATURE_CANDIDATES['sample_feature']
+    # return 'feature_importance' to tuner in first iteration
+    else:
+        sample_col = []
+    df = name2feature(df, sample_col)

    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)

+    # send final result to Tuner
+    nni.report_final_result({
+        "default":val_score , 
+        "feature_importance":feature_imp
    })
```

**define a search space**

```python
{
    "1-order-op" : [
            col1,
            col2
        ],
    "2-order-op" : [
        [
            col1,
            col2
        ], [
            col3, 
            col4
        ]
    ]
}
```

**Get configure from Tuner**

```python
...
RECEIVED_PARAMS = nni.get_next_parameter()
if 'sample_feature' in RECEIVED_PARAMS.keys():
            sample_col = RECEIVED_PARAMS['sample_feature']
else:
    sample_col = []
# raw_feature + sample_feature
df = name2feature(df, sample_col)
...
```

**Send final metric and feature importances to tuner**

```python
feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
nni.report_final_result({
    "default":val_score , 
    "feature_importance":feature_imp
})
```

**Extend the SDK of feature engineer method**

**Run expeirment**

```python
nnictl create --config config.yml
```

### 3.1.3 NNI自动工程测试样例

#### [UCI Heart Dataset](http://archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29)

![image.png](https://i.loli.net/2021/01/03/iuYoDJ2KnvzOdSV.png)

**Data Set Information:**

Cost Matrix

_______ abse pres
absence 0 1
presence 5 0

where the rows represent the true values and the columns the predicted.

**Attribute Information:**
Attribute Information:
\------------------------
-- 1. age
-- 2. sex
-- 3. chest pain type (4 values)
-- 4. resting blood pressure
-- 5. serum cholesterol in mg/dl
-- 6. fasting blood sugar > 120 mg/dl
-- 7. resting electrocardiographic results (values 0,1,2)
-- 8. maximum heart rate achieved
-- 9. exercise induced angina
-- 10. oldpeak = ST depression induced by exercise relative to rest
-- 11. the slope of the peak exercise ST segment
-- 12. number of major vessels (0-3) colored by flourosopy
-- 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

Attributes types
\-----------------

Real: 1,4,5,8,10,12
Ordered:11,
Binary: 2,6,9
Nominal:7,3,13

Variable to be predicted
\------------------------
Absence (1) or presence (2) of heart disease

#### 加载数据集

```python
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import nni
import logging
import numpy as np
import pandas as pd
import json
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.append('../../')

from fe_util import *
from model import *

logger = logging.getLogger('auto-fe-examples')

if __name__ == '__main__':
    file_name = ' ./breast-cancer.data'
    target_name = 'Class'
    id_index = 'Id'

    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()
    logger.info("Received params:\n", RECEIVED_PARAMS)
    
    # list is a column_name generate from tuner
    df = pd.read_csv(file_name, sep = ',')
    df.columns = [
        'Class', 'age', 'menopause', 'tumor-size', 'inv-nodes',
        'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'
    ]
    df['Class'] = LabelEncoder().fit_transform(df['Class'])
    
    if 'sample_feature' in RECEIVED_PARAMS.keys():
        sample_col = RECEIVED_PARAMS['sample_feature']
    else:
        sample_col = []
    
    # raw feaure + sample_feature
    df = name2feature(df, sample_col, target_name)
    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
    nni.report_final_result({
        "default":val_score, 
        "feature_importance":feature_imp
    })
```

**AutoFETuner代码**

```python
# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import copy
import json
import logging
import random
import numpy as np
from itertools import combinations

from enum import Enum, unique

from nni.tuner import Tuner
from nni.utils import extract_scalar_reward, OptimizeMode

from const import FeatureType, AGGREGATE_TYPE

logger = logging.getLogger('autofe-tuner')


class AutoFETuner(Tuner):
    def __init__(self, optimize_mode = 'maximize', feature_percent = 0.6):
        """Initlization function
        count : 
        optimize_mode : contains "Maximize" or "Minimize" mode.
        search_space : define which features that tuner need to search
        feature_percent : @mengjiao
        default_space : @mengjiao 
        epoch_importance : @mengjiao
        estimate_sample_prob : @mengjiao
        """
        self.count = -1
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.search_space = None
        self.feature_percent = feature_percent
        self.default_space = []
        self.epoch_importance = []
        self.estimate_sample_prob = None

        logger.debug('init aufo-fe done.')


    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial graph config, as a serializable object.
        parameter_id : int
        """
        self.count += 1
        if self.count == 0:
            return {'sample_feature': []}
        else:
            sample_p = np.array(self.estimate_sample_prob) / np.sum(self.estimate_sample_prob)
            sample_size = min(128, int(len(self.candidate_feature) * self.feature_percent))
            sample_feature = np.random.choice(
                self.candidate_feature, 
                size = sample_size, 
                p = sample_p, 
                replace = False
                )
            gen_feature = list(sample_feature)
            r = {'sample_feature': gen_feature}
            return r  


    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
        Record an observation of the objective function
        parameter_id : int
        parameters : dict of parameters
        value: final metrics of the trial
        '''
        # get the default feature importance

        if self.search_space is None:
            self.search_space = value['feature_importance']
            self.estimate_sample_prob = self.estimate_candidate_probility()
        else:
            self.epoch_importance.append(value['feature_importance'])
            # TODO
            self.update_candidate_probility()
        reward = extract_scalar_reward(value)
        if self.optimize_mode is OptimizeMode.Minimize:
            reward = -reward

        logger.info('receive trial result is:\n')
        logger.info(str(parameters))
        logger.info(str(reward))
        return


    def update_search_space(self, data):
        '''
        Input: data, search space object.
        {
            'op1' : [col1, col2, ....]
            'op2' : [col1, col2, ....]
            'op1_op2' : [col1, col2, ....]
        }
        '''

        self.default_space = data
        self.candidate_feature = self.json2space(data)


    def update_candidate_probility(self):
        """
        Using true_imp score to modify candidate probility.
        """
        # get last importance
        last_epoch_importance = self.epoch_importance[-1]
        last_sample_feature = list(last_epoch_importance.feature_name)
        for index, f in enumerate(self.candidate_feature):
            if f in last_sample_feature:
                score = max(float(last_epoch_importance[last_epoch_importance.feature_name == f]['feature_score']), 0.00001)
                self.estimate_sample_prob[index] = score
        
        logger.debug("Debug UPDATE ", self.estimate_sample_prob)


    def estimate_candidate_probility(self):
        """
        estimate_candidate_probility use history feature importance, first run importance.
        """
        raw_score_dict = self.impdf2dict()
        logger.debug("DEBUG feature importance\n", raw_score_dict)

        gen_prob = []
        for i in self.candidate_feature:
            _feature = i.split('_')
            score = [raw_score_dict[i] for i in _feature if i in raw_score_dict.keys()]
            if len(score) == 1:
                gen_prob.append(np.mean(score))
            else:
                generate_score = np.mean(score) * 0.9 # TODO
                gen_prob.append(generate_score)
        return gen_prob


    def impdf2dict(self):
        return dict([(i,j) for i,j in zip(self.search_space.feature_name, self.search_space.feature_score)])


    def json2space(self, default_space):
        """
        parse json to search_space 
        """
        result = []
        for key in default_space.keys():
            if key == FeatureType.COUNT:
                for i in default_space[key]:
                    name = (FeatureType.COUNT + '_{}').format(i)
                    result.append(name)         
            
            elif key == FeatureType.CROSSCOUNT:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        if i == j:
                            continue
                        cross = [i,j] 
                        cross.sort()
                        name = (FeatureType.CROSSCOUNT + '_') + '_'.join(cross)
                        result.append(name)         
                        
            
            elif key == FeatureType.AGGREGATE:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        for stat in AGGREGATE_TYPE:
                            name = (FeatureType.AGGREGATE + '_{}_{}_{}').format(stat, i, j)
                            result.append(name)
            
            elif key == FeatureType.NUNIQUE:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        name = (FeatureType.NUNIQUE + '_{}_{}').format(i, j)
                        result.append(name)
            
            elif key == FeatureType.HISTSTAT:
                for i in default_space[key][0]:
                    for j in default_space[key][1]:
                        name = (FeatureType.HISTSTAT + '_{}_{}').format(i, j)
                        result.append(name)
            
            elif key == FeatureType.TARGET:
                for i in default_space[key]:
                    name = (FeatureType.TARGET + '_{}').format(i)
                    result.append(name) 
            
            elif key == FeatureType.EMBEDDING:
                for i in default_space[key]:
                    name = (FeatureType.EMBEDDING + '_{}').format(i)
                    result.append(name) 
            
            else:
                raise RuntimeError('feature ' + str(key) + ' Not supported now')
        return result
```

**实验结果**







