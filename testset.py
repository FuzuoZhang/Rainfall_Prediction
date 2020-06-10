#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd


def main():
    paths = ['station_371_clean.csv','station_372_clean.csv', 'station_373_clean.csv', 'station_374_clean.csv', 'station_375_clean.csv']
    if not os.path.exists('testset'):
        os.makedirs('testset')

    for i in range(5):
        path = paths[i]
        data = pd.read_csv(os.path.join('temp/', path))
        N = data.shape[0]
        n_test = round(N * 0.9)
    
        test = data.iloc[n_test: ]
        print(test.shape)
        test.to_csv('./testset/station'+str(i+1)+'.csv')

if __name__ == '__main__':
    main()  