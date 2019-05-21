# coding: utf-8

import pandas as pd
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn import metrics

fp = 'D:\深圳杯比赛\数据\\test.csv'

with open(fp, 'r', encoding='gb18030') as f:
    df_origin = pd.read_csv(f, index_col=0)

features = df_origin.values
y_pred = SpectralClustering(n_clusters=2, gamma=1).fit_predict(features)
print("Calinski-Harabasz Score with gamma=", 1, "n_clusters=", 2,"score:", metrics.calinski_harabaz_score(features, y_pred))

for i in range(4):
    plot_features = features[:, i:i + 2]
    plt.subplot(2, 2, i + 1), plt.scatter(plot_features[:, 0], plot_features[:, 1], c=y_pred)

plt.show()
