#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd


def main():
    paths = ['station_371.csv','station_372.csv', 'station_373.csv', 'station_374.csv', 'station_375.csv']
    for i in range(5):
        path = paths[i]
        data = pd.read_csv(path)
        N = data.shape[0]
        n_test = round(N * 0.9)

        test = data.iloc[n_test: ]
        print(test.shape)
        test.to_csv('../test/station'+str(i+1)+'.csv')

if __name__ == '__main__':
    main()