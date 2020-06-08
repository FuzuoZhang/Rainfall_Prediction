#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from data_process import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import argparse

def train(train_path, test_path, classifier, balanced, voting):
    global y_result, model
    features_to_remove = ['mdct', 'wsid', 'wsnm', 'elvt', 'lat', 'lon', 'inme', 'city', 'prov','date']
    T = 6
    X_train, y_train, X_valid, y_valid = preprocess(train_path, T, features_to_remove)

    ###### training ######
    if classifier == 'lr' and balanced == False:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        model = lr
        y_result = lr.predict(X_train)
        print('lr training done:')

    if classifier == 'lr' and balanced == True:
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')  # classweight 按照样本数比例生成权重，解决样本不均衡问题
        lr.fit(X_train, y_train)
        model = lr
        y_result = lr.predict(X_train)
        print('lr(balanced) training done:')

    if classifier == 'svm' and balanced == False:
        svc = SVC()
        svc.fit(X_train, y_train)
        model = svc
        y_result = svc.predict(X_train)
        print('svm training done:')

    if classifier == 'svm' and balanced == True:
        svc = SVC(class_weight='balanced')
        svc.fit(X_train, y_train)
        model = svc
        y_result = svc.predict(X_train)
        print('svm(balanced) training done:')

    if classifier == 'knn':
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        model = knn
        y_result = knn.predict(X_train)
        print('knn training done:')
        # grid_paras = {
        #     'n_neighbors': [30, 100]}  # , 'weights':['uniform','distance'], 'metric':['euclidean', 'manhattan'] }
        # gs = GridSearchCV(KNeighborsClassifier(), grid_paras, verbose=1, cv=3, n_jobs=-1)
        # gs.fit(X_train, y_train)
        # y_result = gs.predict(X_train)
        #
        # print("best paras:", gs.best_params_)

    if classifier == 'dt' and balanced == False:
        dtc = DecisionTreeClassifier(max_depth=10)
        dtc.fit(X_train, y_train)
        model = dtc
        y_result = dtc.predict(X_train)
        print('dt training done:')

    if classifier == 'dt' and balanced == True:
        dtc = DecisionTreeClassifier(max_depth=10, class_weight='balanced')
        dtc.fit(X_train, y_train)
        model = dtc
        y_result = dtc.predict(X_train)
        print('dt(balanced) training done:')

    if classifier == 'rf':
        rf = RandomForestClassifier(max_depth=8)
        rf.fit(X_train, y_train)
        model = rf
        y_result = rf.predict(X_train)
        print('rf training done:')

    if classifier == 'gbdt':
        gbdt = GradientBoostingClassifier()
        gbdt.fit(X_train, y_train)
        model = gbdt
        y_result = gbdt.predict(X_train)
        print('gbdt training done:')

    if classifier == 'mlp':
        mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500)
        mlp.fit(X_train, y_train)
        model = mlp
        y_result = mlp.predict(X_train)
        print('mlp training done:')

    if classifier == 'ensemble':
        clf1 = LogisticRegression(max_iter=1000)
        clf2 = SVC(probability=True)
        clf3 = KNeighborsClassifier()
        clf4 = RandomForestClassifier(max_depth = 8)
        clf5 = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=500)
        eclf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('knn', clf3), ('rf', clf4), ('mlp', clf5)],
                                voting=voting)
        clf1 = clf1.fit(X_train, y_train)
        clf2 = clf2.fit(X_train, y_train)
        clf3 = clf3.fit(X_train, y_train)
        clf4 = clf4.fit(X_train, y_train)
        clf5 = clf5.fit(X_train, y_train)
        eclf = eclf.fit(X_train, y_train)

        model = eclf

        y_result = eclf.predict(X_train)
        print('ensemble training done:')

    print(classification_report(y_train, y_result))


    ###### testing ######
    test = feature_selection(test_path, features_to_remove)
    X_test, y_test = datatrans(test, T)
    y_test = value2class(y_test)
    X_test = X_test.reshape(X_test.shape[0], -1)

    sr_X = StandardScaler()
    sr_X = sr_X.fit(X_test)
    X_test = sr_X.transform(X_test)
    y_result = model.predict(X_test)
    print('test done:')
    print(classification_report(y_test, y_result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Classifier.py')
    parser.add_argument('--data-train', type=str, default='./temp/station_371.csv', help='train data file path')
    parser.add_argument('--data-test', type=str, default='./test/station1.csv', help='test data file path')
    parser.add_argument('--classifier', type=str, default='lr', help='classifier name')
    parser.add_argument('--voting', type=str, default='hard', help='hard or soft')
    parser.add_argument('--balanced', action='store_true', help='class weights balanced')
    opt = parser.parse_args()
    print(opt)

    train(opt.data_train, opt.data_test, opt.classifier,opt.balanced, opt.voting)



