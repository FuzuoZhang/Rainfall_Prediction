#!/usr/bin/python
# -*- coding:utf-8 -*-
# Reference: https://www.kaggle.com/sanjayroberts1/exploratory-data-analysis-and-clean-up#Clean-up-entire-dataset
import numpy as np
import pandas as pd

def main():
    # clean-up entire dataset
    df = pd.read_csv('sudeste.csv')

    # prcp 降水量，gbrd太阳能 用0替代nan
    df['prcp'].fillna(0, inplace=True)
    df['gbrd'].fillna(0, inplace=True)

    # Drop where all sensor columns are 0
    col = ['prcp', 'stp', 'smax', 'smin', 'gbrd', 'temp',
           'dewp', 'tmax', 'dmax', 'tmin', 'dmin', 'hmdy', 'hmax', 'hmin', 'wdsp',
           'wdct', 'gust']
    df = df[(df[col] != 0).any(axis=1)]

    # 用0替代nan，线性插值
    df['temp'].replace(0, np.nan, inplace=True)
    df['temp'].interpolate('linear', inplace=True, limit_direction='both')
    df['dewp'].replace(0, np.nan, inplace=True)
    df['dewp'].interpolate('linear', inplace=True, limit_direction='both')
    df['tmax'].replace(0, np.nan, inplace=True)
    df['tmax'].interpolate('linear', inplace=True, limit_direction='both')
    df['tmin'].replace(0, np.nan, inplace=True)
    df['tmin'].interpolate('linear', inplace=True, limit_direction='both')
    df['dmax'].replace(0, np.nan, inplace=True)
    df['dmax'].interpolate('linear', inplace=True, limit_direction='both')
    df['dmin'].replace(0, np.nan, inplace=True)
    df['dmin'].interpolate('linear', inplace=True, limit_direction='both')
    df['hmax'].replace(0, np.nan, inplace=True)
    df['hmax'].interpolate('linear', inplace=True, limit_direction='both')
    df['hmin'].replace(0, np.nan, inplace=True)
    df['hmin'].interpolate('linear', inplace=True, limit_direction='both')
    df['hmdy'].replace(0, np.nan, inplace=True)
    df['hmdy'].interpolate('linear', inplace=True, limit_direction='both')
    df['wdsp'].interpolate('linear', inplace=True, limit_direction='both')
    df['gust'].interpolate('linear', inplace=True, limit_direction='both')
    df['stp'].replace(0, np.nan, inplace=True)
    df['stp'].interpolate('linear', inplace=True, limit_direction='both')
    df['smax'].replace(0, np.nan, inplace=True)
    df['smax'].interpolate('linear', inplace=True, limit_direction='both')
    df['smin'].replace(0, np.nan, inplace=True)
    df['smin'].interpolate('linear', inplace=True, limit_direction='both')

    # 纠正错误的station经纬度
    df['elvt'].replace(0, 4, inplace=True)
    df['lat'].replace(0, -23.993611, inplace=True)
    df['lon'].replace(0, -46.256389, inplace=True)

    # 处理两个有问题的station
    df.loc[df['wsnm'] == 'RIO CLARO', 'lat'] = -22.722778
    df.loc[df['wsnm'] == 'RIO CLARO', 'lon'] = -44.135833
    df.loc[df['wsnm'] == 'SÃO GONÇALO', 'lat'] = -22.826944
    df.loc[df['wsnm'] == 'SÃO GONÇALO', 'lon'] = -43.053889

    # 选择5个气象站并清洗数据
    stations = []
    for id in [371, 372, 373, 374, 375]:
        is_id = df['wsid'] == id
        station = df[is_id]
        stations.append(station)
        station = station.reset_index(drop=True)
        station.to_csv('./temp/station_'+str(id)+'.csv', index = False)

    # clean.csv
    station_371_clean = pd.concat([stations[0][:42800], stations[0][44300:67500], stations[0][75500:]])
    station_371_clean = station_371_clean.reset_index(drop=True)
    station_371_clean.to_csv('./temp/station_371_clean.csv', index = False)

    temp1 = stations[1][:4100]
    temp2 = stations[1][4200:10000]
    temp3 = stations[1][23200:37000]
    temp4 = stations[1][47000:65000]
    temp5 = stations[1][68000:76000]
    temp6 = stations[1][77000:]
    station_372_clean = pd.concat([temp1, temp2, temp3, temp4, temp5, temp6])
    station_372_clean = station_372_clean.reset_index(drop=True)
    station_372_clean.to_csv('./temp/station_372_clean.csv', index = False)

    temp1 = stations[2][3000:7300]
    temp2 = stations[2][19500:41800]
    temp3 = stations[2][43700:54400]
    temp4 = stations[2][56800:85500]
    temp5 = stations[2][88500:]
    station_373_clean = pd.concat([temp1, temp2, temp3, temp4, temp5])
    station_373_clean = station_373_clean.reset_index(drop=True)
    station_373_clean.to_csv('./temp/station_373_clean.csv', index = False)

    temp1 = stations[3][:50000]
    temp2 = stations[3][56800:61300]
    temp3 = stations[3][68800:]
    station_374_clean = pd.concat([temp1, temp2, temp3])
    station_374_clean = station_374_clean.reset_index(drop=True)
    station_374_clean.to_csv('./temp/station_374_clean.csv', index = False)

    station_375_clean = stations[4].reset_index(drop=True)
    station_375_clean.to_csv('./temp/station_375_clean.csv', index = False)


if __name__ == '__main__':
    main()