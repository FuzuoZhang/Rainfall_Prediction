import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from data_process import *
from config import *
import argparse

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


def main(train_path, test_path, classifier):
    X_train, y_train, X_valid, y_valid = preprocess(train_path, T, features_to_remove)
    if classifier == 'xgboost':
        model = xgboost_classifier_train(xgb_params, X_train, y_train)
    if classifier == 'xgboost-bagging':
        model = xgb_bagging_train(X_train, y_train, n_classifiers, xgb_bagging_params)

    test = feature_selection(test_path, features_to_remove)
    X_test, y_test = datatrans(test, T)
    y_test = value2class(y_test)
    X_test = X_test.reshape(X_test.shape[0], -1)

    sr_X = StandardScaler()
    sr_X = sr_X.fit(X_test)
    X_test = sr_X.transform(X_test)
    
    if classifier == 'xgboost':
        xgboost_classifier_predict(model, X_test, y_test)
    if classifier == 'xgboost-bagging':
        xgb_bagging_test(X_test, y_test, model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='xgb_classifier.py')
    parser.add_argument('--data_train', type=str, default='./temp/station_373_clean.csv', help='train data file path')
    parser.add_argument('--data_test', type=str, default='./testset/station3.csv', help='test data file path')
    parser.add_argument('--classifier', type=str, default='xgboost-bagging', help='classifier name')
    opt = parser.parse_args()
    print(opt)

    main(opt.data_train, opt.data_test, opt.classifier)
    '''
    xgboost_main()
    #xgb_bagging_main()
    '''
    
    

    
    
