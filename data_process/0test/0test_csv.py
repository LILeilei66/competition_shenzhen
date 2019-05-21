# coding: utf-8

import pandas as pd

fp = 'D:\深圳杯比赛\数据\工作簿1.csv'
with open(fp,'r', encoding='utf-8') as f:
    df = pd.read_csv(f)

# print(df)
df2 = df.copy()
for column in df.columns:
    if 'Unnamed' in column:
        df2 = df.drop(column, axis=1, inplace=False)

for i in df.keys():
    print(i)