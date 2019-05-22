# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class DrawRador():
    """
    特征column:
        environment, 0:5, = [主城区环境空气细颗粒物（微克/立方米）,每万人生活垃圾清运量（吨）,建成区绿化覆盖率(%),每万人公园绿地面积(公顷）,每万人拥有公厕（座）]
        society, 5:9, = [城镇基本医疗保险年末参保人数比（%）,每万人等级教练员(人),工伤保险年末参保人数比(%),养老保险基金支出比(%)]
        service, 9:14, = [每万名女性妇幼保健院个数(个),每千人医疗卫生机构数（个）,每万人中医院个数（个）,每万人拥有卫生技术人员数(人),每万人卫生机构床位数（张),
                        地方财政医疗卫生支出占财政支出比率(%)]
        people, 14, = [人口死亡率]
        culture, 15, = [每万人社会服务机构(个)]
    """
    def __init__(self):
        self.surfixes = ['数据_健康总.csv']
        self.cities = ['上海','北京', '天津','重庆']
        self.root_dir = '../第一问数据'
        path_weights = '各城市特征权重.csv'
        assert os.path.isfile(path_weights)
        self.index = ['environment', 'society', 'service', 'people', 'culture']
        self.city_list = ['Shanghai', 'Beijing', 'Tianjing', 'Chongqin']

        try:
            self.weights = pd.read_csv(path_weights, index_col=0, encoding='gbk')
        except UnicodeDecodeError:
            self.weights = pd.read_csv(path_weights, index_col=0, encoding='utf-8')

    def main(self):
        for surfix in self.surfixes:
            for j, city in enumerate(self.cities):
                fp = os.path.join(self.root_dir, city + surfix)
                assert os.path.isfile(fp)
                try:
                    df = pd.read_csv(fp, index_col=0, encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(fp, index_col=0, encoding='utf-8')
                for i in range(len(df)):
                    results = self.cal_results(city, df.iloc[i]) # results = [environment, society,
                    # service, people, culture]
                    self.draw_result(results, self.city_list[j], df.iloc[i].name)

    def cal_results(self, city, item):
        """
        通过 city 作为 key 得到权重, 对应 df 中 value 计算得到每一项的结果.
        :param city:
        :param df:
        :return: [environment, society, service, people, culture]
        """
        note_environment = np.matmul(self.weights.loc[city][:5].values, item[:5].values)
        note_society = np.matmul(self.weights.loc[city][5:9].values, item[5:9].values)
        note_service = np.matmul(self.weights.loc[city][9:14].values, item[9:14].values)
        note_people = np.multiply(self.weights.loc[city][14], item[14])
        note_culture = np.multiply(self.weights.loc[city][15], item[15])
        print([note_environment / 100, note_society, note_service, note_people, note_culture])
        return [note_environment / 100, note_society, note_service, note_people, note_culture]

    def draw_result(self, results, city, year):
        results = np.hstack((results, results[0]))  # 使图像闭合
        angles = np.linspace(0, 2*np.pi, num=len(results)-1, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, results, 'o-', linewidth=2)
        ax.fill(angles, results, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, self.index)
        ax.set_ylim(min(results), max(results) * 1.1)
        plt.title('{:} of year {:}'.format(city, year))
        ax.grid(True)
        plt.savefig('{:} of year {:}'.format(city, year))
        plt.show()

if __name__ == '__main__':
    draw_rador = DrawRador()
    draw_rador.main()
