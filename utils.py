import random
import time

import faiss
import h5py
import numpy as np
import scipy
import torch.nn.functional as F
from faiss import normalize_L2
from numpy.core import multiarray
from scipy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def generate_pair(y, label_cell_indx, num, error_rate=0):
    """
    Generate pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = np.array(y)

    for i in range(len(label_cell_indx)):
        for j in range(len(label_cell_indx)):
            if y[label_cell_indx[i]] == y[label_cell_indx[j]]:
                ml_ind1.append(label_cell_indx[i])
                ml_ind2.append(label_cell_indx[j])
            else:
                cl_ind1.append(label_cell_indx[i])
                cl_ind2.append(label_cell_indx[j])

    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


def generate_random_pair(y, label_cell_indx, num, n_clusters, error_rate=0):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = np.array(y)

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
            if ind1 == l1 and ind2 == l2:
                return True
        return False

    error_num = 0
    num0 = num

    arr = np.empty((210, n_clusters+1))
    multiarray.copyto(arr, -1, casting='unsafe')
    k =0
    for i in label_cell_indx:
        # print( int(y[label_cell_indx[i]]))
        arr[k, int(y[i])] = 1
        k+=1;
    # while num > 0:

    # 构建G
    x_num = arr.shape[0]
    y_num = arr.shape[1]
    row = []
    line = []
    g = np.zeros((210, 210))

    # constraint_a
    for i in range(y_num):
        for j in range(x_num):
            for k in range(x_num):
                if (arr[j, i] == arr[k, i] and arr[j, i] == 1) or j == k:
                    g[j, k] = 1
                    g[k, j] = 1
    ## constraint_b
    for i in range(y_num):
        for j in range(x_num):
            if arr[j, i] == 1:
                row.append(j)
                line.append(i)

    num = len(row)

    for m in range(num):
        for n in range(num):
            if line[m] != line[n]:
                g[row[m], row[n]] = -1
                g[row[n], row[m]] = -1

    for i in range(y_num):
        for j in range(x_num):
            for k in range(x_num):
                if (arr[j, i] == 1 and arr[k, i] == -1) or (arr[j, i] == -1 and arr[k, i] == 1):
                    g[j, k] = -1
                    g[k, j] = -1

    print("G:", g)
    while num0 > 0:
        tmp1 =  random.randint(0, 209)
        tmp2 =  random.randint(0, 209)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if g[tmp1][tmp2] == 1:
            if error_num >= error_rate * num0:
                ml_ind1.append(label_cell_indx[tmp1])
                ml_ind2.append(label_cell_indx[tmp2])
            else:
                cl_ind1.append(label_cell_indx[tmp1])
                cl_ind2.append(label_cell_indx[tmp2])
                error_num += 1
        else:
            if error_num >= error_rate * num0:
                cl_ind1.append(label_cell_indx[tmp1])
                cl_ind2.append(label_cell_indx[tmp2])
            else:
                ml_ind1.append(label_cell_indx[tmp1])
                ml_ind2.append(label_cell_indx[tmp2])
                error_num += 1
        num0 -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)

    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num

def generate_random_pair_new(latent_embedding, y, label_cell_indx, num, n_clusters, error_rate=0,ML=0.1, CL=0.9):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = np.array(y)

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
            if ind1 == l1 and ind2 == l2:
                return True
        return False

    error_num = 0
    num0 = num

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec > 0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)

    arr = np.empty((210, n_clusters+1))
    multiarray.copyto(arr, -1, casting='unsafe')
    k =0
    for i in label_cell_indx:
        # print( int(y[label_cell_indx[i]]))
        arr[k, int(y[i])] = 1
        k+=1;
    # while num > 0:

    # 构建G
    x_num = arr.shape[0]
    y_num = arr.shape[1]
    row = []
    line = []
    g = np.zeros((210, 210))  #

    # constraint_a
    for i in range(y_num):
        for j in range(x_num):
            for k in range(x_num):
                if (arr[j, i] == arr[k, i] and arr[j, i] == 1) or j == k:
                    g[j, k] = 1
                    g[k, j] = 1
    ## constraint_b
    for i in range(y_num):
        for j in range(x_num):
            if arr[j, i] == 1:
                row.append(j)
                line.append(i)

    num = len(row)

    for m in range(num):
        for n in range(num):
            if line[m] != line[n]:
                g[row[m], row[n]] = -1
                g[row[n], row[m]] = -1

    for i in range(y_num):
        for j in range(x_num):
            for k in range(x_num):
                if (arr[j, i] == 1 and arr[k, i] == -1) or (arr[j, i] == -1 and arr[k, i] == 1):
                    g[j, k] = -1
                    g[k, j] = -1

    print("G:", g)
    while num0 > 0:
        tmp1 =  random.randint(0, 209)
        tmp2 =  random.randint(0, 209)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        # if g[tmp1][tmp2] == 1:
        if g[tmp1][tmp2] == 1 and norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            if error_num >= error_rate * num0:
                ml_ind1.append(label_cell_indx[tmp1])
                ml_ind2.append(label_cell_indx[tmp2])
            else:
                cl_ind1.append(label_cell_indx[tmp1])
                cl_ind2.append(label_cell_indx[tmp2])
                error_num += 1
        # elif g[tmp1][tmp2] != 1:
        elif g[tmp1][tmp2] != 1 and norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2)  < cutoff_ML:
            if error_num >= error_rate * num0:
                cl_ind1.append(label_cell_indx[tmp1])
                cl_ind2.append(label_cell_indx[tmp2])
            else:
                ml_ind1.append(label_cell_indx[tmp1])
                ml_ind2.append(label_cell_indx[tmp2])
                error_num += 1
        else:
            continue
        num0 -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    # 随机排序-打乱
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num

def generate_random_pair_from_proteins(latent_embedding, num, ML=0.1, CL=0.9):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
            if ind1 == l1 and ind2 == l2:
                return True
        return False

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec > 0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)

    while num > 0:
        tmp1 = random.randint(0, latent_embedding.shape[0] - 1)
        tmp2 = random.randint(0, latent_embedding.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) < cutoff_ML:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2

def generate_random_pair_from_proteins_new(label_cell_indx,latent_embedding, num, ML=0.1, CL=0.9):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
            if ind1 == l1 and ind2 == l2:
                return True
        return False

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec > 0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)


    while num > 0:
        tmp1 = random.randint(0, latent_embedding.shape[0] - 1)
        tmp2 = random.randint(0, latent_embedding.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) < cutoff_ML:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2

def generate_random_pair_from_CD_markers(markers, num, low1=0.4, high1=0.6, low2=0.2, high2=0.8):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
            if ind1 == l1 and ind2 == l2:
                return True
        return False

    gene_low1 = np.quantile(markers[0], low1)
    gene_high1 = np.quantile(markers[0], high1)
    gene_low2 = np.quantile(markers[1], low1)
    gene_high2 = np.quantile(markers[1], high1)

    gene_low1_ml = np.quantile(markers[0], low2)
    gene_high1_ml = np.quantile(markers[0], high2)
    gene_low2_ml = np.quantile(markers[1], low2)
    gene_high2_ml = np.quantile(markers[1], high2)
    gene_low3 = np.quantile(markers[2], low2)
    gene_high3 = np.quantile(markers[2], high2)
    gene_low4 = np.quantile(markers[3], low2)
    gene_high4 = np.quantile(markers[3], high2)

    while num > 0:
        tmp1 = random.randint(0, markers.shape[1] - 1)
        tmp2 = random.randint(0, markers.shape[1] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if markers[0, tmp1] < gene_low1 and markers[1, tmp1] > gene_high2 and markers[0, tmp2] > gene_high1 and markers[
            1, tmp2] < gene_low2:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        elif markers[0, tmp2] < gene_low1 and markers[1, tmp2] > gene_high2 and markers[0, tmp1] > gene_high1 and \
                markers[1, tmp1] < gene_low2:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        elif markers[1, tmp1] > gene_high2_ml and markers[2, tmp1] > gene_high3 and markers[1, tmp2] > gene_high2_ml and \
                markers[2, tmp2] > gene_high3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[1, tmp1] > gene_high2_ml and markers[2, tmp1] < gene_low3 and markers[1, tmp2] > gene_high2_ml and \
                markers[2, tmp2] < gene_low3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] > gene_high3 and markers[1, tmp2] > gene_high1_ml and \
                markers[2, tmp2] > gene_high3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] < gene_low3 and markers[3, tmp1] > gene_high4 and \
                markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] < gene_low3 and markers[3, tmp2] > gene_high4:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] < gene_low3 and markers[3, tmp1] < gene_low4 and \
                markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] < gene_low3 and markers[3, tmp2] < gene_low4:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


def generate_random_pair_from_embedding_clustering(latent_embedding, num, n_clusters, ML=0.005, CL=0.8):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
            if ind1 == l1 and ind2 == l2:
                return True
        return False

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit(latent_embedding).labels_

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec > 0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)

    while num > 0:
        tmp1 = random.randint(0, latent_embedding.shape[0] - 1)
        tmp2 = random.randint(0, latent_embedding.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if y_pred[tmp1] == y_pred[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif y_pred[tmp1] != y_pred[tmp2] and norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


def readhdf5():

    f = h5py.File('/home/dhu133/ZHC/scDCC/data/CITE_PBMC/CITE_CBMC_counts_top2000.h5', 'r')  # 打开h5文件

    for key in f.keys():
        print(key)
        print(f[key].name)
        print(f[key].shape)
        # print(f[key])
    print(f['X'][:])
    return None


def mask_generator(p_m, x):
    """Generate mask vector.

  Args:
    - p_m: corruption probability
    - x: feature matrix

  Returns:
    - mask: binary mask matrix
  """
    mask = np.random.binomial(1, p_m, x.shape)
    # mask = x
    return mask


def pretext_generator(m, x):
    """Generate corrupted samples.

  Args:
    m: mask matrix
    x: feature matrix

  Returns:
    m_new: final mask matrix after corruption
    x_tilde: corrupted feature matrix
  """

    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    # print(input_logits,target_logits)
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.mse_loss(input_softmax, target_softmax, size_average=False)

def sortAndIndex(x, y):
    genes = x.shape[1]
    x1 = x.cpu()
    # y = y.cpu()
    x_y = np.insert(x1, x1.shape[1], values=y, axis=1)
    x_y1 = x_y[x_y[:, x1.shape[1]].argsort()]
    index = x_y[:, x1.shape[1]].argsort().numpy()
    print(index)
    x1 = x_y1[:, 0:genes]
    # y1 = y
    # y1.sort()
    return x1, index


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

