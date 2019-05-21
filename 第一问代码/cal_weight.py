# encoding: gbk
from utils.common import CalFunctions
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

HEALTH_TYPES = ['人群', '服务', '环境', '社会'] #, '文化'] 深圳数据表无文化
HEALTH_TYPES_EN = ['people', 'service', 'environment', 'society']
FILE_PREFIXES = ['深圳数据_v1_健康']
FILE_PREFIXES_EN = ['Shenzhen_v1_health']
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

class HealthEvaluation():
    """
    对于深圳市的人群, 服务, 环境, 社会, 文化,
    通过 CalFunctions.entropy_weight 进行计算
    """
    def __init__(self):
        pass

    @staticmethod
    def cal_weight(fp):
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
        weights = w.values.reshape(1, 5)[0]
        weights[0], weights[1] = - weights[0], - weights[1]
        df['note'] = -1
        # print(w)

        for i in range(len(df)):
            item = df.iloc[i, :-1]
            fts = item.values
            note = np.matmul(weights, fts)
            df.iloc[i, -1] = note

        print(df)

    def load_weights(self):
        self.weight_people = np.array([0.164449, -0.262902, -0.155697, -0.190223, -0.226728])
        # [平均预期寿命  人口死亡率     婴儿死亡率  5岁以下儿童死亡率  孕产妇死亡率]
        self.weight_service = np.array([0.254562, 0.244977, 0.132770, 0.135699, 0.231992])
        # [每万名女性妇幼保健院个数(个）  每千人医疗卫生机构数（个）  每万人中医院个数（个）  每万人医生（人)  每万人卫生机构床位数（张)]
        self.weight_environment = np.array([-0.096855, -0.111615, 0.663590, 0.051602, 0.076337])
        # [灰霾日  每万人生活垃圾清运量  建成区绿化覆盖率  每万人公园绿地面积   每万人拥有公厕]
        self.weight_society = np.array([0.0425028, 0.574972])
        # [每万人二级运动员（人）  每万人社会养老机构年末收养人数（人）]

    def cal_note_from_fp(self, fp):
        # 根据 key 获得 weight_category
        try:
            df = pd.read_csv(fp, index_col=0, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(fp, index_col=0, encoding='utf-8')
        df.fillna(-1)
        print('df:\n', df)

        note_people = np.matmul(self.weight_people, df.iloc[0,:5].values)
        note_service = np.matmul(self.weight_service, df.iloc[0,5:10].values)
        note_environment = np.matmul(self.weight_environment, df.iloc[0,10:15].values)
        note_society = np.matmul(self.weight_society, df.iloc[0,15:17].values)
        print(MSG_NOTES.format(note_people, note_service, note_environment, note_society))

        values = [note_people, note_service, note_environment, note_service, note_people]
        # 使图像闭合
        angles = np.linspace(0, 2*np.pi, num=4, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, HEALTH_TYPES)
        ax.set_ylim(0, max(values) * 1.1)
        plt.title(FILE_PREFIXES[0])
        ax.grid(True)
        plt.show()

    def cal_note_from_df(self, df,i,title, fig=None):
        print('df:\n', df)

        note_people = np.matmul(self.weight_people, df[:5].values)
        note_service = np.matmul(self.weight_service, df.iloc[5:10].values)
        note_environment = np.matmul(self.weight_environment, df.iloc[10:15].values)
        note_society = np.matmul(self.weight_society, df.iloc[15:17].values)
        print(MSG_NOTES.format(note_people, note_service, note_environment, note_society))

        values = [note_people, note_service, note_environment, note_service, note_people]
        # 使图像闭合
        angles = np.linspace(0, 2*np.pi, num=4, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(1,2,i, polar=True)
        # ax.title(title)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, HEALTH_TYPES_EN)
        ax.set_ylim(0, max(values) * 1.1)
        plt.title(df.name)
        ax.grid(True)
        return fig

    def cal_note_2(self, df1, df2):
        print('df1:\n', df1)
        print('df2:\n', df2)

        note_people = np.matmul(self.weight_people, df1[:5].values)
        note_service = np.matmul(self.weight_service, df1.iloc[5:10].values)
        note_environment = np.matmul(self.weight_environment, df1.iloc[10:15].values)
        note_society = np.matmul(self.weight_society, df1.iloc[15:17].values)
        print(MSG_NOTES.format(note_people, note_service, note_environment, note_society))
        values1 = [note_people, note_service, note_environment, note_service, note_people]
        # 使图像闭合

        note_people = np.matmul(self.weight_people, df2[:5].values)
        note_service = np.matmul(self.weight_service, df2.iloc[5:10].values)
        note_environment = np.matmul(self.weight_environment, df2.iloc[10:15].values)
        note_society = np.matmul(self.weight_society, df2.iloc[15:17].values)
        print(MSG_NOTES.format(note_people, note_service, note_environment, note_society))
        values2 = [note_people, note_service, note_environment, note_service, note_people]

        angles = np.linspace(0, 2 * np.pi, num=4, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, polar=True)
        ax.plot(angles, values1, 'o-', linewidth=2, label=df1.name)
        ax.fill(angles, values1, alpha=0.25)
        ax.plot(angles, values2, 'o-', linewidth=2, label=df2.name)
        ax.fill(angles, values2, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, HEALTH_TYPES_EN)
        ax.set_ylim(0, max(max(values1), max(values2)) * 1.1)
        plt.title('Shenzhen_v1')
        plt.legend(loc='best')
        ax.grid(True)
        plt.show()


    def cal_note_3(self, df1, df2, df3):
        print('df1:\n', df1)
        print('df2:\n', df2)
        print('df3:\n', df3)

        # <editor-fold desc="df1">
        note_people = np.matmul(self.weight_people, df1[:5].values)
        note_service = np.matmul(self.weight_service, df1.iloc[5:10].values)
        note_environment = np.matmul(self.weight_environment, df1.iloc[10:15].values)
        note_society = np.matmul(self.weight_society, df1.iloc[15:17].values)
        print(MSG_NOTES.format(note_people, note_service, note_environment, note_society))
        values1 = [note_people, note_service, note_environment / 3, note_service, note_people]
        # </editor-fold>
        # 使图像闭合

        # <editor-fold desc="df2">
        note_people = np.matmul(self.weight_people, df2[:5].values)
        note_service = np.matmul(self.weight_service, df2.iloc[5:10].values)
        note_environment = np.matmul(self.weight_environment, df2.iloc[10:15].values)
        note_society = np.matmul(self.weight_society, df2.iloc[15:17].values)
        print(MSG_NOTES.format(note_people, note_service, note_environment, note_society))
        values2 = [note_people, note_service, note_environment / 3, note_service, note_people]
        # </editor-fold>

        # <editor-fold desc="df3">
        note_people = np.matmul(self.weight_people, df3[:5].values)
        note_service = np.matmul(self.weight_service, df3.iloc[5:10].values)
        note_environment = np.matmul(self.weight_environment, df3.iloc[10:15].values)
        note_society = np.matmul(self.weight_society, df3.iloc[15:17].values)
        print(MSG_NOTES.format(note_people, note_service, note_environment, note_society))
        values3 = [note_people, note_service, note_environment / 3, note_service, note_people]
        # </editor-fold>

        angles = np.linspace(0, 2 * np.pi, num=4, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, polar=True)
        ax.plot(angles, values1, 'o-', linewidth=2, label=df1.name)
        ax.fill(angles, values1, alpha=0.25)
        ax.plot(angles, values2, 'o-', linewidth=2, label=df2.name)
        ax.fill(angles, values2, alpha=0.25)
        ax.plot(angles, values3, 'o-', linewidth=2, label=df3.name)
        ax.fill(angles, values3, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, HEALTH_TYPES_EN)
        ax.set_ylim(min(min(values1), min(values2), min(values3)) * 0.9, \
                    max(max(values1), max(values2), max(values3)) * 1.1)
        plt.title('Shenzhen_v1')
        plt.legend(loc='best')
        ax.grid(True)
        plt.show()

if __name__ == '__main__':
    health_evaluation = HealthEvaluation()
    health_evaluation.load_weights()
    fp = os.path.join(FILE_ROOT, '深圳数据_v1.csv')
    try:
        df = pd.read_csv(fp, index_col=0, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(fp, index_col=0, encoding='utf-8')

    health_evaluation.cal_note_3(df.iloc[0], df.iloc[1], df.iloc[2])



    """health_evaluation = HealthEvaluation()
    health_evaluation.load_weights()
    fp = os.path.join(FILE_ROOT, '深圳数据_v1.csv')
    try:
        df = pd.read_csv(fp, index_col=0, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(fp, index_col=0, encoding='utf-8')

    fig = plt.figure()

    for i in range(len(df)):
        item = df.iloc[i]
        fig = health_evaluation.cal_note_from_df(df=item, i=i+1, title='深圳数据_v1', fig=fig)
    plt.show()"""

    """health_evaluation = HealthEvaluation()
    health_evaluation.load_weights()
    fp_2017 = os.path.join(FILE_ROOT, '深圳数据_v1_2017.csv')
    health_evaluation.cal_note_from_fp(fp_2017)"""


    """for file_prefix in FILE_PREFIXES:
        for health_type in HEALTH_TYPES:
            print('\n==={:}====='.format(health_type))
            fp = os.path.join(FILE_ROOT, file_prefix + health_type + '.csv')
            HealthEvaluation.cal_weight(fp)"""
