conda 新建环境
```conda create -n rainfall python=3.7```

激活环境
```conda activate rainfall```

安装需要的包:
numpy == 1.18.1
pandas == 0.25.3
scikit-learn == 0.23.1
argparse = 1.4.0
```conda install numpy pandas scikit-learn```

下载原始数据的代码：
从https://cloud.tsinghua.edu.cn/d/a96c9fb8f56d4fb5be62/ 下载hourly-weather-surface.zip 解压到codes文件夹，得到sudeste.csv数据。

清洗数据：
```python data_washing.py```
在codes/temp文件下得到371-375五个气象站的完整数据和进一步清洗的数据，共10个csv文件。其中进一步清洗指的是对于某些一段时间缺失的特征，直接丢弃对应的数据条。

训练模型，并在测试集上得到结果：
```python Classifier.py --data-train ./temp/station_371.csv --data-test ./test/station5.csv --classifier svm --balanced -voting soft```
一共有5个超参数：

- --data-train：训练数据地址，默认为./temp/station_371.csv
- --data-test: 测试数据地址，默认为./test/station5.csv，可以尝试不同的组合
- --classifier：共有lr，svm，knn, dt, rf, gbdt, mlp, ensemble等八种选择
- --balanced：输入balanced则表示true，否则false。只有lr, svm, dt有balanced选项。
- --voting：只有ensemble有这个参数，值有hard和soft，对应不同的投票方式

得到结果以lr为例：

```python
python Classifier.py --data-test ./test/station2.csv --classifier lr --balanced 
Namespace(balanced=True, classifier='lr', data_test='./test/station2.csv', data_train='./temp/station_371.csv', voting='hard')

lr(balanced) training done:
              precision    recall  f1-score   support

           0       0.98      0.81      0.89     85835
           1       0.18      0.53      0.27      4892
           2       0.17      0.34      0.23      1529
           3       0.08      0.59      0.14       595
    
    accuracy                           0.79     92851
   macro avg       0.35      0.57      0.38     92851
weighted avg       0.92      0.79      0.84     92851

test done:
              precision    recall  f1-score   support

           0       0.99      0.78      0.87      9887
           1       0.16      0.56      0.25       525
           2       0.15      0.36      0.21       197
           3       0.05      0.51      0.08        45
    
    accuracy                           0.76     10654
   macro avg       0.33      0.55      0.35     10654
weighted avg       0.93      0.76      0.82     10654
```

