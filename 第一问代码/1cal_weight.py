# encoding: gbk
from utils.common import CalFunctions
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import os
import pandas as pd

# <editor-fold desc="string info">
HEALTH_TYPES_EN = ['people', 'service', 'environment', 'society']
FILE_PREFIXES = ['北京数据_v4_健康'] # '上海数据_v1_健康'] # '重庆数据_v1_健康'] # '天津数据_v1_健康']'北京数据_v4_健康']
HEALTH_TYPES = ['人群', '服务', '环境', '社会', '文化'] # 深圳数据表无文化
FILE_PREFIXES_EN = ['Shenzhen_v4_health']
# NEGATIVE_FEATURES = ['人口死亡率(‰)', '婴儿死亡率(‰)', '5岁以下儿童死亡率(‰)', '孕产妇死亡率(1/10万)', \ # 服务
#                      '灰霾日（天）', '每万人生活垃圾清运量（吨）', \ # 环境
#                      '每万人社会养老机构年末收养人数（人）' \ # 社会
#                      ]
MSG_NOTES = """note_people: {:};
note_service: {:};
note_environment: {:};
note_society: {:}.
"""
FILE_ROOT = '../第一问数据'
# </editor-fold>

class HealthEvaluation():
    """
    对于深圳市的人群, 服务, 环境, 社会, 文化,
    通过 CalFunctions.entropy_weight 进行计算
    """
    def __init__(self):
        pass

    @staticmethod
    def cal_weight(fp):
        assert os.path.isfile(fp)
        try:
            df = pd.read_csv(fp, index_col=0, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(fp, index_col=0, encoding='utf-8')
        df.fillna(-1)
        print('df:\n', df)
        w = CalFunctions.entropy_weight(df)  # 调用cal_weight
        print('w:\n',w)
        w.index = df.columns
        w.columns = ['weight']
        weights = w.values.reshape(1, len(w))[0]
        # weights[0], weights[1] = - weights[0], - weights[1]
        df['note'] = -1
        print(w)

        for i in range(len(df)):
            item = df.iloc[i, :-1]
            fts = item.values
            note = np.matmul(weights, fts)
            df.iloc[i, -1] = note

        print(df)


    @staticmethod
    def cal_weights_all(fp):
        """对于每一个城市，他的特征维度为二，第一个维度为年份，(此后称为维度Y, Year); 第二个维度为特征(此后称为维度F, Features).
        1. 对于整个特征矩阵，沿维度Y进行归一化 (使的每个特征在其的年份范围内，都在(0,1)区间中);
        2. 对于整个矩阵通过其特征的熵求取系数;
        3. 对于每个年份的所有特征进行加权求和作为当年健康评价分数.
        """
        # <editor-fold desc="0. Read Dataframe">
        assert os.path.isfile(fp)
        try:
            df = pd.read_csv(fp, index_col=0, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(fp, index_col=0, encoding='utf-8')
        df.fillna(-1)
        print('df:\n', df)
        # </editor-fold>

        # <editor-fold desc="1. Data Normalization">
        # 记录 max || min
        """
                Feature1   ...
        Year1   value1     ...
        ...
        MIN     value1     ...
        MAX     value1     ...
        WEIGHT  value1     ...
        """

        df_normalized = copy(df.iloc[:-2])
        min = df.iloc[-2]
        max = df.iloc[-1]
        for i in range(len(df) - 2):
            item = df.iloc[i]
            df_normalized.iloc[i] = (item - min) / (max - min)
        print(df_normalized)
        # </editor-fold>

        w = CalFunctions.entropy_weight(df_normalized)  # 调用cal_weight
        w.columns = df_normalized.columns
        w.index = ['WEIGHTS']
        print('w:\n', w)
        with open(fp, 'a') as f:
            w.to_csv(f, header=False)

class Main():
    @staticmethod
    def cal_weight():
        for file_prefix in FILE_PREFIXES:
            for health_type in HEALTH_TYPES:
                print('\n==={:}====='.format(health_type))
                fp = os.path.join(FILE_ROOT, file_prefix + health_type + '.csv')
                HealthEvaluation.cal_weight(fp)

    @staticmethod
    def  cal_weights():
        fp = 'F:\\2018-2019-2\深圳杯\第一问数据\\上海数据_健康总_copy.csv'
        HealthEvaluation.cal_weights_all(fp)

if __name__ == '__main__':
    Main.cal_weights()


    """health_evaluation = HealthEvaluation()
    health_evaluation.load_weights()
    fp_2017 = os.path.join(FILE_ROOT, '深圳数据_v1_2017.csv')
    health_evaluation.cal_note_from_fp(fp_2017)"""


