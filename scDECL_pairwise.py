import os
from time import time

import h5py
import numpy as np
import scanpy.api as sc
import torch
from sklearn import metrics

from preprocess import read_dataset, normalize
from scDECL import scDECL
from utils import cluster_acc, generate_random_pair

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=16, type=int)
    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--label_cells_files', default='label_selected_cells_1.txt')
    parser.add_argument('--n_pairwise', default=5000, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file',
                        default='data/Large_Datasets/Macosko_mouse_retina.h5')
    parser.add_argument('--datasets', default='worm')
    parser.add_argument('--maxiter', default=100, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDCC_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')
    parser.add_argument('--alpha', default=2, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--mu', default=1, type=float)
    parser.add_argument('--pm', default=0.5, type=float)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    data_mat = h5py.File(args.data_file)
    print(data_mat.keys())
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])

    args.n_clusters = np.unique(y).shape[0]
    print("n_clusters:",args.n_clusters)
    data_mat.close()
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata1 = normalize(adata,
                       size_factors=True,
                       normalize_input=True,
                       logtrans_input=True,
                       highly_genes=None)
    input_size = adata1.n_vars

    print(args)

    print(adata1.X.shape)
    print(y.shape)

    if not os.path.exists(args.label_cells_files):
        indx = np.arange(len(y))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells * len(y)))]
    else:
        label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

    x_sd = adata1.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    if args.n_pairwise > 0:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num = generate_random_pair(y, label_cell_indx, args.n_pairwise,args.n_clusters,args.n_pairwise_error)
        print("Must link paris: %d" % ml_ind1.shape[0])
        print("Cannot link paris: %d" % cl_ind1.shape[0])
        print("Number of error pairs: %d" % error_num)
    else:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])

    sd = 2.5

    model = scDECL(input_dim=adata1.n_vars, z_dim=32, n_clusters=args.n_clusters,
                  encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma,
                  ml_weight=args.ml_weight, cl_weight=args.ml_weight).cuda()
    print(str(model))

    t0 = time()

    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata1.X, X_raw=adata1.raw.X, size_factor=adata.obs.size_factors,
                                   batch_size=args.batch_size, epochs=args.pretrain_epochs,
                                   ae_weights=args.ae_weight_file, alpha=args.alpha, beta=args.beta,
                                   datasets=args.datasets, pm=args.pm, mu=args.mu)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    y_pred, _, _, _, _ = model.fit(X=adata1.X, X_raw=adata1.raw.X, sf=adata1.obs.size_factors, y=y,
                                   batch_size=args.batch_size, num_epochs=args.maxiter,
                                   ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                                   update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)

    print('Total time: %d seconds.' % int(time() - t0))

    eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    eval_cell_y = np.delete(y, label_cell_indx)

    acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    if not os.path.exists(args.label_cells_files):
        np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")
