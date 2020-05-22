### 降雨量预测—数据集分析

巴西122个气象站从2000-2016的气象数据。见sudeste.csv，共9779168 lines * 31 cols。 

[数据来源]:https://www.kaggle.com/PROPPG-PPG/hourly-weather-surface-brazil-southeast-region

具体参数（按顺序）：

|  参数名    | 含义     |数据类型      |  备注    |
| ---- | ---- | ---- | ---- |
|   1 wsid   |   气象站id   | int      |      |
|   2 wsnm   |  气象站名字    |  string    |  不重要    |
|   3 elva   |   海拔   |    float  |      |
| 4 lat | 纬度 | float |      |
| 5 lon | 经度 | float |      |
| 6 inme | 气象站number | string | 不重要 |
| 7 city | 城市 | string |      |
| 8 prov | 省份 | string |      |
| 9 mdct | 日期+时间（每小时） | String | 2007-11-06 00：00：00 |
| 10 yr | 年份 | int | 2000-2016 |
| 11 mo | 月份 | int | 0-12 |
| 12 da | 日期 | int | 0-31 |
| 13 hr | 小时 | int | 0-23 |
| 14 prcp | 过去一小时降水量mm | float | NAN |
| 15 stp | 实时每小时的气压hPa(小数点后一位，下同) | float |      |
| 16 smax | 过去一小时最高气压hPa | float |      |
| 17 smin | 过去一小时最低气压hPa | float |  |
| 18 gbrd | 太阳能KJ/m2 | float | NAN |
| 19 temp | 实时温度 C | float |      |
| 20 dewp | 露点实时温度C | float |      |
| 21 tmax | 过去一小时最高温度 C | float |      |
| 22 dmax | 过去一小时最高露点温度 C | float |      |
| 23 tmin | 过去一小时最低温度 C | float |      |
| 24 dmin | 过去一小时最低露点温度 C | float |      |
| 25 hmdy | 实时相对湿度 % | float |      |
| 26 hmax | 过去一小时最高相对湿度 % | float |      |
| 27 hmin | 过去一小时最低相对湿度 % | float |      |
| 28 wdsp | 风速 m/s | float |      |
| 29 wdct | 风向度数 | float | 0-360 |
| 30 gust | 阵风 m/s | float |      |

