import pandas as pd
import numpy as np
from alipy.query_strategy import QueryInstanceUncertainty
from sklearn.metrics import accuracy_score
from Function.data import preProcess
def ac_entropy(a,b,unlab,test, model,data, test_data):
    # 模型训练由训练集决定，标注工作在训练集中完成
    model.fit(a,b)
    X,y = preProcess(data)
    # ac是由新数据集来决定的
    X_test, y_test = preProcess(test_data)
    ac = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))
    #ac = accuracy_score(y_true=y[test], y_pred=model.predict(X[test, :]))
    strategy = QueryInstanceUncertainty()
    # 预测unlab池内的可能性
    prob = model.predict_proba(X[unlab])
    # 计算熵值大小
    entropy = strategy.calc_entropy(prob)
    list = []
    for i in unlab:
        list.append(i)

    entropy = pd.DataFrame({'sample_index': list, 'class': y[list], 'entropy': entropy}).sort_values(by='entropy',ascending=False)
    entropy.loc[:, 'rank'] = np.arange(0, len(unlab))
    return ac, entropy

def accuracy_show(lab,test,model,data):
    X, y = preProcess(data)
    model.fit(X[lab, :], y[lab])
    validation_ac = accuracy_score(y_true=y[test], y_pred=model.predict(X[test, :]))
    return validation_ac
def entropy_show(model,X_lab,y_lab, X_unlab, udata):
    strategy = QueryInstanceUncertainty()
    # 通过新的lab标签来训练模型
    model.fit(X_lab,y_lab)
    # 预测unlab池内的可能性
    prob = model.predict_proba(X_unlab)
    # 计算熵值大小
    entropy = strategy.calc_entropy(prob)
    matrix = pd.concat([pd.DataFrame({'entropy':entropy}), udata],axis = 1 ).sort_values(by='entropy',ascending=False)

    entropy = pd.DataFrame({'entropy':entropy}).sort_values(by='entropy',ascending=False)
    print(matrix)


    return entropy, matrix
def get_video2(data,index):
    list = []
    for i in index:
        video = data.loc[i, ['video']].values[0]
        if video in [0,3,6,9]:
            list.append('0403')
        elif video in [1, 4, 7, 10]:
            list.append('0608')
        elif video in [12, 15, 18, 21]:
            list.append('0713')
        elif video in [13, 16, 19, 22]:
            list.append('0720')
    return list

def initial_show(model,data):
    from Function.index_handle import separate
    train, test, lab, unlab,ni,ma = separate(data)
    X,y = preProcess(data)
    model.fit(X[lab, :], y[lab])
    initial_accuracy = accuracy_score(y_true=y[test], y_pred=model.predict(X[test, :]))
    return initial_accuracy, model