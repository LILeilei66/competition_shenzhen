# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits.mplot3d import axes3d


import time
import warnings
warnings.filterwarnings('ignore')

fp = 'D:\深圳杯比赛\数据\\上海_processed.csv'

with open(fp, 'r', encoding='utf-8') as f:
    df_origin = pd.read_csv(f, index_col=0)

df_origin = df_origin.fillna(value=-1)

features = df_origin.values

def spectral_cluster():
    scores =[]
    for k in (2, 3, 4, 5):
        for gamma in (0.1,0.3,0.9, 1, 2, 3, 4, 5,6,7,8,9):
            y_pred = SpectralClustering(n_clusters=k, gamma=gamma, n_neighbors=5).fit_predict(features)
            scores.append(metrics.calinski_harabaz_score(features, y_pred))
            print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(features, y_pred))
            if metrics.calinski_harabaz_score(features, y_pred) > 6:
                break
        if scores[-1] > 6:
            break
    print(np.max(scores))

    # y_pred = SpectralClustering(n_clusters=5, gamma=0.4, n_neighbors=6).fit_predict(features)
    # print("Calinski-Harabasz Score with gamma=", 0.4, "n_clusters=", 5,"score:", metrics.calinski_harabaz_score(features, y_pred))

    """
    for i in range(features.shape[1]):
        for j in range(features.shape[1]):
            # plot_features = features[:, i:i + 2]
            plt.scatter(features[:, i], features[:, j], c=y_pred)
            plt.show()
            time.sleep(1)
    """
    return y_pred

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.scatter(data[:,0], data[:,1], c=label), plt.show()

if __name__ == '__main__':
    # Calinski-Harabasz Score with gamma= 9 n_clusters= 2 score: 7.6041930900214805
    # 7.6041930900214805
    y_pred = spectral_cluster()
    tsne = TSNE(n_components=3)
    result = tsne.fit_transform(features)
    # plot_embedding(result, y_pred,
    #                      't-SNE embedding of the digits')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # 创建一个三维的绘图工程

    ax.scatter(result[:,0], result[:,1], result[:, 2], c=y_pred)
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

