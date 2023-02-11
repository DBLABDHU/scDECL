import numpy as np
import h5py
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    plt.figure(figsize=(12, 10))
    # 为每个条形图添加数值标签
#     plt.title('Worm neuron cells with DCA')
    scatterColors = ['purple', 'blue', 'green', 'red', 'orange', 'yellow', 'brown', 'black', 'grey', 'pink']
    p = []
    q = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            # if clusterRes[j] == i:
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        p.append(plt.scatter(x1, y1, s=80, c=color, alpha=1, marker='.'))
    # plt.legend(p, q, loc='upper left')
    plt.savefig(c[0], dpi=1000)
    plt.show()

def cluster_acc(y_true, y_pred):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

a = ["D:/idea-lab/scDCC/data/SMILR/mouse.txt",
     "D:/idea-lab/scDCC/data/SMILR/worm.txt",
     "D:/idea-lab/scDCC/data/CITE_PBMC/CITE_CBMC_counts_top2000.h5",
     "D:/idea-lab/scDCC/data/Large_Datasets/Macosko_mouse_retina.h5",
     "D:/idea-lab/scDCC/data/Small_Datasets/human_kidney_select_2100_top2000.h5",
     "D:/idea-lab/scDCC/data/Small_Datasets/mouse_bladder_cell_select_2100_top2000.h5",
     "D:/idea-lab/scDCC/data/Small_Datasets/worm_neuron_cell_select_2100_top2000.h5",
     "D:/idea-lab/scDCC/data/Small_Datasets/10X_PBMC_select_2100_top2000.h5",
     ]
b = [16, 10]
c = ['D:/idea-lab/scDCC/results/SMILR_Mouse_tsne_pred_notitle1.jpg',
     'D:/idea-lab/scDCC/results/SMILR_Worm_tsne_pred_notitle1.jpg',
     'D:/idea-lab/scDCC/results/555.jpg'
     ]
e = ["D:/idea-lab/scDCC/data/SMILR/mouse_label.txt",
     "D:/idea-lab/scDCC/data/SMILR/worm_label.txt"]
# Y=true label, Y1=scgnn label, Y2=CIDR label, Y3=SMILR label
# x = np.loadtxt(a[0],dtype=float)
# # y_pred = np.loadtxt(d[1], dtype=int)
# # y = np.array(data_mat['Y3'])
# y = np.loadtxt(e[0],dtype=int)
# y_pred = np.array(data_mat['Y2'])
import umap

data_mat = h5py.File(a[7])
print(data_mat.keys())
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])
# embedding = np.array(data_mat['ADT_X'])
# print('emmmm',embedding)
n_clusters = np.unique(y).shape[0]
print("n_clusters:", n_clusters)
data_mat.close()
# data1 = umap.UMAP(n_neighbors=16, min_dist=0.9, n_components=2).fit_transform(x)
from sklearn.decomposition import PCA
data1 = TSNE(n_components=2).fit_transform(x)
data1 = PCA(n_components=2).fit_transform(data1)

y_pred1 = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(data1)
# plotRes(data1, y, b[1])

# 计算
acc = np.round(cluster_acc(y, y_pred1), 5)
nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred1), 5)
ari = np.round(metrics.adjusted_rand_score(y, y_pred1), 5)
print('result '+': ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))