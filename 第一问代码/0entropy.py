import pandas as pd
import numpy as np
import math

fp = '../第一问数据/深圳数据_v1_健康环境.csv'

with open(fp, 'r', encoding='utf-8') as f:
    df = pd.read_csv(fp, index_col=0, encoding='utf-8')
df.fillna(-1)

def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]

    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = np.array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = np.array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj  # 计算各样本的综合得分,用最原始的数据

    w = pd.DataFrame(w)
    return w

w = cal_weight(df)  # 调用cal_weight
w.index = df.columns
w.columns = ['weight']
weights = w.values.reshape(1, 5)[0]
weights[0], weights[1] = - weights[0],  - weights[1]
df['note'] = -1
# print(w)

for i in range(len(df)):
    item = df.iloc[i,:-1]
    fts = item.values
    note = np.matmul(weights, fts)
    df.iloc[i, -1] = note

print(df)