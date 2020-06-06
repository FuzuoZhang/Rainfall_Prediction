T = 6
features_to_remove = ['mdct','wsid','wsnm','elvt','lat','lon','inme','city','prov','date']
#features_to_remove = ['mdct','wsid','wsnm','elvt','lat','lon','inme','city','prov','date',
#                       'smax','smin','tmax','tmin','dmax','dmin','hmax','hmin']

xgb_params =  {'max_depth':8, 
            'learning_rate':0.01,
            'n_estimators':300,
            'booster':'gbtree', 
            'nthread':-1,
            'gamma':0.1,
            'subsample':0.8,
            'colsample_bytree':0.7,
            'colsample_bylevel':1,
            'silent':False, 
            'reg_alpha':0,
            'reg_lambda':1, 
            'min_child_weight':3,
            'scale_pos_weight':1,
            'objective':'multi:softmax',
            'num_class':4}


xgb_bagging_params = {'max_depth':5, 
     'learning_rate':0.01,
     'n_estimators':5,
     'booster':'gbtree', 
     'nthread':-1,
     'gamma':0.1,
     'silent':False, 
     'reg_alpha':0,
     'reg_lambda':1,
     'objective':'multi:softmax',
     'num_class':4}


sta371 = 'station_371.csv'
sta372 = 'station_372.csv'
sta373 = 'station_373.csv'
sta374 = 'station_374.csv'
sta375 = 'station_375.csv'
stations = [sta371, sta372, sta373, sta374, sta375]
stations_clean = []
for i in range(5):
    stations_clean.append(stations[i].split('.')[0]+"_clean.csv")
    