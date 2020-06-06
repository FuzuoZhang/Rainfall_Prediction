import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from data_process import *
from config import *


import warnings
warnings.filterwarnings("ignore")


#XGboost
def xgboost_classifier_train(params, X_train, y_train):
    clf = xgb.XGBClassifier(params = params)
    clf.fit(X_train, y_train)
    return clf

def xgboost_classifier_predict(clf, X, y):
    y_pre = clf.predict(X)
    print("正确率：{}".format(accuracy_score(y, y_pre)))
    print("分类结果报告：\n", classification_report(y,y_pre))


def xgboost(data, params):
    X_train, y_train, X_valid, y_valid, X_test, y_test = data
    xgb_clf = xgboost_classifier_train(params, X_train, y_train)
    
    print('\n/*******训练集实验结果**********/')
    xgboost_classifier_predict(xgb_clf, X_train, y_train)
    
    print('\n/*******验证集实验结果**********/')
    xgboost_classifier_predict(xgb_clf, X_valid, y_valid)
    
    print('\n/*******测试集实验结果**********/')
    xgboost_classifier_predict(xgb_clf, X_test, y_test)


def xgb_bagging_train(X_train, y_train, n_classifiers, params):
    n = X_train.shape[0]
    forest = []
    
    for i in range(n_classifiers):
        rand_ind = np.random.randint(n,size=n)
        subX_train = X_train[rand_ind]
        suby_train = y_train[rand_ind]
        clf = xgboost_classifier_train(params, subX_train, suby_train)
        forest.append(clf)
        
    return forest

def xgb_bagging_test(X_test, y_test, models):
    n_classifiers = len(models)
    n = X_test.shape[0]
    pre_Y = np.zeros((n,n_classifiers), dtype='int')
    for i in range(n_classifiers):
        pre_Y[:,i] = models[i].predict(X_test)
    y_pre= np.zeros((n,), dtype='int')
    for i in range(n):
        count_label = Counter(pre_Y[i,:])
        y_pre[i] = max(count_label, key=count_label.get)
    
    print("正确率：{}".format(accuracy_score(y_test, y_pre)))
    print("分类结果报告：\n", classification_report(y_test,y_pre))


def xgb_bagging(path, params, n_classifiers):
    X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_xgboost(path, T)
    
    models = xgb_bagging_train(X_train, y_train, n_classifiers, params=params)
    
    print('\n/*******训练集实验结果**********/')
    xgb_bagging_test(X_train, y_train, models)
    
    print('\n/*******验证集实验结果**********/')
    xgb_bagging_test(X_valid, y_valid, models)
    
    print('\n/*******测试集实验结果**********/')
    xgb_bagging_test(X_test, y_test, models)
    

def xgboost_main():
    #在每一个气象站数据上分别训练xgboost模型，并用验证集、测试集测试模型性能
    print("\n/*********Xgboost分类器**********/\n")
    print("paramters: \n", xgb_params, "\n")
    
    for i in range(5):
        print(stations[i].split('.')[0])
        path = os.path.join('data/', stations_clean[i])
        data = preprocess_xgboost(path, T, features_to_remove)
        xgboost(data, xgb_params)
        print('\n\n')


def xgb_bagging_main():
    #在每一个气象站数据上分别训练bagging-xgboost模型，并用验证集、测试集测试模型性能

    n_classifiers = 10
    print('\n/*********Xgboost 随机森林**********/')
    print("paramters: \n", xgb_bagging_params, "\n")

    for i in range(5):
        print(stations[i].split('.')[0])
        path = os.path.join('data/', stations_clean[i])
        data = preprocess_xgboost(path, T, features_to_remove)
        xgb_bagging(path, xgb_bagging_params, n_classifiers)
        print('\n\n')
    

if __name__ == "__main__":
    xgboost_main()
    #xgb_bagging_main()
    
    

    
    
