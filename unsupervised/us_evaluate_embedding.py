import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
'''
1.class sklearn.model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)[source]
    n_splitsint，默认值 = 5
    折叠数。必须至少为 2。
    shuffle布尔值，默认值=假
    是否在拆分为批次之前随机排列每个类的样本。 请注意，每个拆分中的样本不会被随机排列。
    random_stateint、随机状态实例或无，默认值 = 无 当为 True 时，会影响 索引，用于控制每个类的每个折叠的随机性。
     否则，保留为 。 传递一个 int 以获得跨多个函数调用的可重现输出。 请参阅词汇表。shufflerandom_staterandom_stateNone
StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同

2.class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None,
  verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
  
（1）estimator选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法：estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10),
（2）param_grid
    需要最优化的参数的取值，值为字典或者列表，例如：param_grid =param_test1，param_test1 = {'n_estimators':range(10,71,10)}。
（3）scoring=None
    模型评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。具体值的选取看本篇第三节内容。
（4）fit_params=None
（5） n_jobs=1
    n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值
（6） iid=True
    iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。
（7） refit=True
    默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。
（8）cv=None
    交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。
（9）verbose=0, scoring=None
    verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
（10）pre_dispatch=‘2*n_jobs’
    指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次
（11）error_score=’raise’
（12）return_train_score=’warn’
    如果“False”，cv_results_属性将不包括训练分数
回到sklearn里面的GridSearchCV，GridSearchCV用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数

'''
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

from IPython import embed

def draw_plot(datadir, DS, embeddings, fname, max_nodes=None):
    return
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    print('fitting TSNE ...')
    x = TSNE(n_components=2).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])

    df['x0'], df['x1'], df['Y'] = x[:,0], x[:,1], y
    sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=5)
    plt.legend()
    plt.savefig(fname)

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    
    accuracies = []
    for train_index, test_index in kf.split(x, y):


        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}#优化的参数取值
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)

        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

    accuracies = np.array(accuracies)
    return accuracies.mean(), accuracies.std()

def evaluate_embedding(embeddings, labels, search=True):

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    acc = 0
    acc_val = 0

    _acc_val, _acc = svc_classify(x,y, search)
    if _acc_val > acc_val:
        acc_val = _acc_val
        acc = _acc

    return acc_val, acc
