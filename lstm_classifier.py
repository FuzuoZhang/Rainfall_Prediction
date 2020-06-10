import os
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from data_process import *
from config import *
import argparse


def standscale(X_train, X_valid, X_test):
    T = X_train.shape[1]

    X_train = X_train.reshape(X_train.shape[0],-1)
    X_valid = X_valid.reshape(X_valid.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0],-1)

    # normalization
    sr_X = StandardScaler()
    sr_X = sr_X.fit(X_train)
    X_train = sr_X.transform(X_train)
    X_valid = sr_X.transform(X_valid)
    X_test = sr_X.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], T, -1)
    X_valid = X_valid.reshape(X_valid.shape[0], T, -1)
    X_test = X_test.reshape(X_test.shape[0], T, -1)

    return X_train, X_valid, X_test


# LSTM网络
class LSTM_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM_Classifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.f = nn.Sequential(nn.Linear(hidden_size*6, output_size), 
                               nn.Softmax())
        
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        output, _ = self.rnn(x)
        seq_len, batch, hidden_size = output.shape
        #output = self.f(output[-1,:,:].view(-1, hidden_size))
        output = self.f(output.view(-1, hidden_size*6))
        return output 


def lstm_train(X_train, y_train, hidden_size=10, num_layers=2, epochs=30, lr=0.01):
    #数据格式处理
    n_train, t, p = X_train.shape
    #n_valid = X_valid.shape[0]
    X_train = X_train.reshape(-1,n_train,p)
    #X_valid = X_valid.reshape(-1,n_valid,p)
    
    #转为张量
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train) 
   
    #X_valid = torch.from_numpy(X_valid).float()
    
    model = LSTM_Classifier(input_size = p, hidden_size = hidden_size, output_size = 4, num_layers=num_layers)
    print(model)
    Loss = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum = 0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for i in range(epochs):
        model.zero_grad() #梯度清零
        var_X = Variable(X_train)#.type(torch.FloatTensor)
        var_y = Variable(y_train)#.type(torch.FloatTensor)
        out = model(var_X)
        #print(out.shape, var_y.shape)
        loss = Loss(out, var_y.long().squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1)%5 ==0:
            print('Epoch: {}, Loss: {:.5f}'.format(i+1, loss.item()))
        
    return model


def lstm_predict(model, X, y):
    n, t, p = X.shape
    X = X.reshape(-1,n,p)
    X = torch.from_numpy(X).float()
    var_X = Variable(X)
    output = model(var_X)
    y_pre = output.data.max(dim=1)[1]
    y_pre = y_pre.numpy()
    print("正确率：{}".format(accuracy_score(y, y_pre)))
    print("分类结果报告：\n", classification_report(y,y_pre))
    return output


def lstm_bagging_train(X_train, y_train, n_classifiers):
    n = X_train.shape[0]

    forest = []
    for i in range(n_classifiers):
        rand_ind = np.random.randint(n,size=n)
        subX_train = X_train[rand_ind]
        suby_train = y_train[rand_ind]
        model = lstm_train(X_train, y_train, hidden_size=10, num_layers=2)
        forest.append(model)
    return forest
             

def lstm_bagging_test(X_test, y_test, models):
    n_classifiers = len(models)
    
    n, t, p = X_test.shape
    pre_Y = np.zeros((n,n_classifiers), dtype='int')

    X = X_test.reshape(-1,n,p)
    X = torch.from_numpy(X).float()
    for i in range(n_classifiers):
        model = models[i]
        var_X = Variable(X)
        output = model(var_X)
        tmp = output.data.max(dim=1)[1]
        pre_Y[:,i] = tmp.numpy()

    y_pre= np.zeros((n,), dtype='int')
    for i in range(n):
        count_label = Counter(pre_Y[i,:])
        y_pre[i] = max(count_label, key=count_label.get)
    
    print("正确率：{}".format(accuracy_score(y_test, y_pre)))
    print("分类结果报告：\n", classification_report(y_test,y_pre))


def easyensemble_train(X_train, y_train, n_classifiers):
    ind_0 = np.where(y_train==0)[0]
    ind_not0 = np.where(y_train!=0)[0]
    X_train_0 = X_train[ind_0]
    y_train_0 = y_train[ind_0]
    X_train_not0 = X_train[ind_not0]
    y_train_not0 = y_train[ind_not0]
    n_is0 = len(ind_0)
    n_not0 = len(ind_not0)

    
    forest = []
    for i in range(n_classifiers):
        rand_ind = np.random.randint(n_is0, size=round(n_not0*0.4))
        sub_X_0 = X_train_0[rand_ind]
        sub_y_0 = y_train_0[rand_ind]
        subX_train = np.concatenate((sub_X_0, X_train_not0), axis=0)
        suby_train = np.concatenate((sub_y_0, y_train_not0), axis=0)
        n_train = len(suby_train)
        shuff_ind = np.arange(n_train)
        np.random.shuffle(shuff_ind)
        subX_train = subX_train[shuff_ind]
        suby_train = suby_train[shuff_ind]
        model = lstm_train(subX_train, suby_train, hidden_size=10, num_layers=2, epochs=200, lr=0.2)
        forest.append(model)
    return forest


def lstm_bagging(data, n_classifiers):
    X_train, y_train, X_valid, y_valid, X_test, y_test = data
    models = lstm_bagging_train(X_train, y_train, n_classifiers)

    print('\n/*******训练集实验结果**********/')
    lstm_bagging_test(X_train, y_train, models)
    
    print('\n/*******验证集实验结果**********/')
    lstm_bagging_test(X_valid, y_valid, models)

    print('\n/*******测试集实验结果**********/')
    lstm_bagging_test(X_test, y_test, models)


def easyensemble(data, n_classifiers):
    X_train, y_train, X_valid, y_valid, X_test, y_test = data
    models = easyensemble_train(X_train, y_train, n_classifiers)

    print('\n/*******训练集实验结果**********/')
    lstm_bagging_test(X_train, y_train, models)
    
    print('\n/*******验证集实验结果**********/')
    lstm_bagging_test(X_valid, y_valid, models)

    print('\n/*******测试集实验结果**********/')
    lstm_bagging_test(X_test, y_test, models)
    

def lstm_bagging_main():
    n_classifiers = 5
    print('\n/*********LSTM 随机森林**********/')
    
    for i in range(5):
        print(stations[i].split('.')[0])
        path = os.path.join('data/', stations_clean[i])
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data2(path, T, features_to_remove)
        X_train, X_valid, X_test = standscale(X_train, X_valid, X_test)
        data = (X_train, y_train, X_valid, y_valid, X_test, y_test)
        lstm_bagging(data, n_classifiers)
        print('\n\n')



def easyensemble_main():
    n_classifiers = 100
    print('\n/*********LSTM easy ensemble**********/')
    
    for i in range(5):
        print(stations[i].split('.')[0])
        path = os.path.join('data/', stations_clean[i])
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data2(path, T, features_to_remove)
        X_train, X_valid, X_test = standscale(X_train, X_valid, X_test)
        data = (X_train, y_train, X_valid, y_valid, X_test, y_test)
        easyensemble(data, n_classifiers)
        print('\n\n')

def main(train_path, test_path):
    X_train, y_train, X_valid, y_valid = load_data(train_path, T, features_to_remove)
    test = feature_selection(test_path, features_to_remove)
    X_test, y_test = datatrans(test, T)
    y_test = value2class(y_test)
    X_train, X_valid, X_test = standscale(X_train, X_valid, X_test)

    model = lstm_train(X_train, y_train)
    lstm_predict(model, X_test, y_test)


if __name__ == "__main__":
    #lstm_bagging_main()
    #easyensemble_main()

    parser = argparse.ArgumentParser(prog='lstm_classifier.py')
    parser.add_argument('--data_train', type=str, default='./temp/station_374_clean.csv', help='train data file path')
    parser.add_argument('--data_test', type=str, default='./testset/station4.csv', help='test data file path')
    opt = parser.parse_args()
    print(opt)

    main(opt.data_train, opt.data_test)