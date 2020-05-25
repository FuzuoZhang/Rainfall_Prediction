## 降雨量预测

目标值：prcp = Amount of precipitation in millimetres (last hour)

### 1. 评价指标

#### MSE（均方误差）

$MSE =\frac {\sum(Y_{actual} - Y_{predict})^2}{N}$

#### RMSE（均方根误差）

用于数据更好的描述。在描述误差时，误差的结果和数据是一个级别的。

$RMSE = \sqrt{MSE}$

#### R-square（决定系数）

$R^2 = 1-\frac{\sum(Y_{actual}-Y_{predict})^2}{\sum(Y_{actual}-Y_{mean})^2}$

决定系数通过数据的变化来表征一个拟合的好坏

- 越接近1，表明变量对y的解释能力越强，模型对数据的拟合也较好
- 越接近0，表明模型拟合的越差
- 经验值：>0.4，拟合效果好
- 缺点：数据集的样本越大，R²越大，因此，不同数据集的模型结果比较会有一定的误差

#### scikit-learn中的各种衡量指标

```python
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score #R square

mean_squared_error(y_test, y_predict)
mean_absolute_error(y_test, y_predict)
r2_score(y_test, y_predict)
```