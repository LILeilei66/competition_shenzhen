# encoding: gbk
from utils.common import CalFunctions
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import os
import pandas as pd

# <editor-fold desc="string info">
HEALTH_TYPES_EN = ['people', 'service', 'environment', 'society']
FILE_PREFIXES = ['��������_v4_����'] # '�Ϻ�����_v1_����'] # '��������_v1_����'] # '�������_v1_����']'��������_v4_����']
HEALTH_TYPES = ['��Ⱥ', '����', '����', '���', '�Ļ�'] # �������ݱ����Ļ�
FILE_PREFIXES_EN = ['Shenzhen_v4_health']
# NEGATIVE_FEATURES = ['�˿�������(��)', 'Ӥ��������(��)', '5�����¶�ͯ������(��)', '�в���������(1/10��)', \ # ����
#                      '�����գ��죩', 'ÿ���������������������֣�', \ # ����
#                      'ÿ����������ϻ�����ĩ�����������ˣ�' \ # ���
#                      ]
MSG_NOTES = """note_people: {:};
note_service: {:};
note_environment: {:};
note_society: {:}.
"""
FILE_ROOT = '../��һ������'
# </editor-fold>

class HealthEvaluation():
    """
    ���������е���Ⱥ, ����, ����, ���, �Ļ�,
    ͨ�� CalFunctions.entropy_weight ���м���
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
        w = CalFunctions.entropy_weight(df)  # ����cal_weight
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
        """����ÿһ�����У���������ά��Ϊ������һ��ά��Ϊ��ݣ�(�˺��Ϊά��Y, Year); �ڶ���ά��Ϊ����(�˺��Ϊά��F, Features).
        1. ������������������ά��Y���й�һ�� (ʹ��ÿ�������������ݷ�Χ�ڣ�����(0,1)������);
        2. ������������ͨ��������������ȡϵ��;
        3. ����ÿ����ݵ������������м�Ȩ�����Ϊ���꽡�����۷���.
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
        # ��¼ max || min
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

        w = CalFunctions.entropy_weight(df_normalized)  # ����cal_weight
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
        fp = 'F:\\2018-2019-2\���ڱ�\��һ������\\�Ϻ�����_������_copy.csv'
        HealthEvaluation.cal_weights_all(fp)

if __name__ == '__main__':
    Main.cal_weights()


    """health_evaluation = HealthEvaluation()
    health_evaluation.load_weights()
    fp_2017 = os.path.join(FILE_ROOT, '��������_v1_2017.csv')
    health_evaluation.cal_note_from_fp(fp_2017)"""


