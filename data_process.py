import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def datatrans(data, T):
    #窗口滑动，每上T时刻的特征和下一时刻的降雨量组成一个样本
    n,p = data.shape    
    all_features = data.columns
    ind=np.where(all_features=='prcp')[0][0]

    n_sample = n-T
    newdata = np.zeros((n_sample,T,p))
    target = np.zeros((n_sample,))
    for i in range(n_sample):
        newdata[i] = data.iloc[i:i+T]
        target[i] = data.iloc[i+T,ind]
    return newdata, target


def value2class(y):
    #y值中的降雨量量化为类别
    y[np.where((y>0)&(y<=1))]=1
    y[np.where((y>1)&(y<=4))]=2
    y[np.where(y>4)]=3
    #y[np.where(y>16)]=4
    return y.astype('int')


def feature_selection(path, features_to_remove=[]):
    data = pd.read_csv(path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    data = data.drop(columns = features_to_remove)
    return data

# features_to_remove = ['mdct','wsid','wsnm','elvt','lat','lon','inme','city','prov']

def load_data(path, T, features_to_remove=[]):
    '''
    data = pd.read_csv(path)
    data = data.drop(columns=['Unnamed: 0'])
    '''
    data = feature_selection(path, features_to_remove)
    # 按照7：2：1的比例拆分训练集、验证集、测试集
    N = data.shape[0]
    n_train = round(N * 0.7)
    n_valid = round(N * 0.2)
    n_test = N - n_train - n_valid

    train = data.iloc[:n_train]
    valid = data.iloc[n_train:n_train + n_valid]
    test = data.iloc[n_train + n_valid:]

    #窗口滑动
    X_train, y_train = datatrans(train, T)
    X_valid, y_valid = datatrans(valid, T)
    X_test, y_test = datatrans(test, T)

    #分类
    y_train = value2class(y_train)
    y_valid = value2class(y_valid)
    y_test = value2class(y_test)
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def preprocess_xgboost(path, T, features_to_remove=[]):
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(path, T, features_to_remove)
    
    #展开
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0],-1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # normalization
    sr_X = StandardScaler()
    sr_X = sr_X.fit(X_train)
    X_train = sr_X.transform(X_train)
    X_valid = sr_X.transform(X_valid)
    X_test = sr_X.transform(X_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


