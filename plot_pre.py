import os
from time import time
from sklearn.manifold import TSNE
import h5py
import numpy as np
import scanpy.api as sc
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from preprocess import read_dataset, normalize
from scDECL import scDECL
from utils import cluster_acc, generate_random_pair


def plotRes(data, clusterRes, clusterNum):
    nPoints = len(data)
    plt.figure(figsize=(12, 10))
    # 为每个条形图添加数值标签
    #     plt.title('Worm neuron cells with DCA')
    scatterColors = ['purple', 'blue', 'green', 'red', 'orange', 'yellow', 'brown', 'black', 'grey', 'pink', 'darkgray',
                     'darkcyan', 'darkblue', 'darkgreen', 'darkred', 'darkorange', 'ivory', 'maroon',
                     'violet', 'darkgrey', 'salmon']
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
    plt.savefig('results/Macosko_mouse_retina.jpg', dpi=1000)
    plt.show()


data_file = 'data/Large_Datasets/Macosko_mouse_retina.h5'
data_mat = h5py.File(data_file)
x = np.array(data_mat['X'])
y = np.array(data_mat['Y'])
data_mat.close()
# n_clusters = np.unique(y).shape[0]
n_clusters = 39
print(n_clusters)
# preprocessing scRNA-seq read counts matrix
adata = sc.AnnData(x)
adata.obs['Group'] = y

adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

adata1 = normalize(adata,
                   size_factors=True,
                   normalize_input=True,
                   logtrans_input=True
                  )

input_size = adata1.n_vars

print(adata1.X.shape)
print(y.shape)

x_sd = adata.X.std(0)
x_sd_median = np.median(x_sd)
print("median of gene sd: %.5f" % x_sd_median)

sd = 2.5

model = scDECL(input_dim=adata1.n_vars, z_dim=32, n_clusters=n_clusters,
              encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=1,
              ml_weight=1, cl_weight=1).cuda()

t0 = time()
ae_weights = 'results/scDCC_p0_1/FTcheckpoint_91.pth.tar'

if os.path.isfile(ae_weights):
    print("==> loading checkpoint '{}'".format(ae_weights))
    checkpoint = torch.load(ae_weights)
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("==> no checkpoint found at '{}'".format(ae_weights))
    raise ValueError

Z, _ = model.encodeBatch(torch.tensor(adata1.X).cuda())
kmeans = KMeans(n_clusters, n_init=20)
y_pred = kmeans.fit_predict(Z.data.cpu().numpy())
data1 = TSNE(n_components=2).fit_transform(Z.cpu())
data1 = PCA(n_components=2).fit_transform(data1)
plotRes(data1, y_pred, n_clusters)
