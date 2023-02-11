import datetime
import math
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

import st_loss

from utils import cluster_acc, mask_generator, pretext_generator


def buildNetwork1(layers, type, activation="relu",dropout = 0.5):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        # net.append(nn.BatchNorm1d(layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

def buildNetwork(layers, type, activation="relu",dropout = 0.8):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        # net.append(nn.BatchNorm1d(layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scDECL(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, encodeLayer=[], decodeLayer=[],
                 activation="relu", sigma=1., alpha=1., gamma=1., ml_weight=1., cl_weight=1.):
        super(scDECL, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.encoder = buildNetwork1([input_dim] + encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer, type="decode", activation=activation)
        self.decoder_mask = buildNetwork([z_dim] + decodeLayer + [input_dim], type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q1 = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q2 = q1 ** ((self.alpha + 1.0) / 2.0)
        q3 = torch.where(torch.isnan(q2), torch.full_like(q2, 0), q2)
        q = (q3.t() / torch.sum(q3, dim=1)).t()
        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forward(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _m = self.decoder_mask(z)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _m

    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        q_all = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, q, _ = self.forward(inputs)
            q_all.append(q.data)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        q_all = torch.cat(q_all, dim=0)
        return encoded, q_all

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)
        return self.gamma * kldloss

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            p = p1 * p2
            a = torch.sum(p, dim=1)
            b = -torch.log(a)
            c = torch.mean(b)
            return self.ml_weight * ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return self.cl_weight * cl_loss

    def costrative_loss(self, p1, p2):
        ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
        return ml_loss

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=600, ae_save=True,
                             ae_weights='AE_weights.pth.tar', alpha=2, beta=1, datasets=None, pm=0.7, mu=1,xishu = 0.85, mixup=0.99):
        criterion_rep = st_loss.SupConLoss(temperature=0.07)
        use_cuda = torch.cuda.is_available()
        print("######", use_cuda, "lr:", lr)
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        now = datetime.datetime.now()
        ae_weights = 'D:/idea-lab/scDECL/results/pretrain/%s_model_%d_%d_%d_%d%.3f.pth.tar' % (
            datasets, now.day, now.hour, now.minute, now.second, pm)
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                # generate mask_X
                m = mask_generator(p_m=0.7, x=x_batch)
                m2 = mask_generator(0.7, x_batch)
                mask, mask_x = pretext_generator(m, x_batch)
                mask2, mask_x2 = pretext_generator(m2, x_batch)
                X_tensor = Variable(mask_x).float().cuda()
                X_tensor2 = Variable(mask_x2).float().cuda()
                X_tensor3 = mixup * X_tensor + (1 - mixup) *  X_tensor2
                mask_tensor = Variable(mask).cuda()
                X_tensor2 = xishu * X_tensor2 + (1-xishu) * X_tensor
                z, _, _m = self.forward(X_tensor)
                z2, _, _ = self.forward(X_tensor2)
                z3, _, _ = self.forward(X_tensor3)
                z_ = torch.nn.functional.normalize(z)
                z2_ = torch.nn.functional.normalize(z2)
                z3_ = torch.nn.functional.normalize(z3)
                mixup_z_ = mixup * z_ + (1 - mixup) * z2_
                features = torch.cat(
                    [z_.unsqueeze(1),
                     z2_.unsqueeze(1)],
                    dim=1)
                cos_loss = criterion_rep.forward(features)

                loss2 = nn.functional.binary_cross_entropy_with_logits(_m.float(), mask_tensor.float())
                loss3 = nn.functional.binary_cross_entropy_with_logits(mixup_z_.float(), z3_.float())

                if(epoch >= 10) :
                    loss = beta * loss2 + cos_loss
                else:
                    loss = beta * loss2 + cos_loss + alpha * loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Pretrain epoch [{}],loss:{:.4f},'.format(epoch + 1, loss.item()))
        if ae_save:
            print(ae_weights)
            torch.save({'ae_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        # now = datetime.datetime.now()
        newfilename = os.path.join(filename,
                                   'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)
        print('newfilename', newfilename)

    def fit(self, X, X_raw, sf, ml_ind1=np.array([]), ml_ind2=np.array([]), cl_ind1=np.array([]), cl_ind2=np.array([]),
            ml_p=1., cl_p=1., y=None, lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir="",
            ):
        '''X: tensor data'''
        criterion_rep = st_loss.SupConLoss(temperature=0.07)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        X = torch.tensor(X).cuda()
        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data, q_all = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        print('loss = cluster_loss + recon_loss + cos_loss')
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        ml_num_batch = int(math.ceil(1.0 * ml_ind1.shape[0] / batch_size))
        cl_num_batch = int(math.ceil(1.0 * cl_ind1.shape[0] / batch_size))
        cl_num = cl_ind1.shape[0]
        ml_num = ml_ind1.shape[0]

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        update_ml = 1
        update_cl = 1

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                # X = Variable(X)
                latent, q_all1 = self.encodeBatch(X)

                q = self.soft_assign(latent)
                # q = Variable(q)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch + 1, acc, nmi, ari))

                # save current model
                if (epoch > 0 and delta_label < tol) or epoch % 10 == 0:
                    self.save_checkpoint({'epoch': epoch + 1,
                                          'state_dict': self.state_dict(),
                                          'mu': self.mu,
                                          'p': p,
                                          'q': q,
                                          'y_pred': self.y_pred,
                                          'y_pred_last': self.y_pred_last,
                                          'y': y
                                          }, epoch + 1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0


            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                xrawbatch = X_raw[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                sfbatch = sf[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch,  _ = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                loss = cluster_loss

                loss.backward()
                optimizer.step()

                cluster_loss_val += cluster_loss.data * len(inputs)
                train_loss = cluster_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f " % (
                epoch + 1, train_loss / num, cluster_loss_val / num))

            ml_loss = 0.0
            if epoch % update_ml == 0:

                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw1 = X_raw[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf1 = sf[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    px2 = X[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf2 = sf[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw2 = X_raw[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    rawinputs1 = Variable(pxraw1)
                    sfinput1 = Variable(sf1)
                    inputs2 = Variable(px2)
                    rawinputs2 = Variable(pxraw2)
                    sfinput2 = Variable(sf2)
                    z1, q1, _ = self.forward(inputs1)
                    z2, q2, _ = self.forward(inputs2)
                    loss = (ml_p * self.pairwise_loss(q1, q2, "ML"))

                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()


            cl_loss = 0.0
            if epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    px1 = X[cl_ind1[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    px2 = X[cl_ind2[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    z1, q1, _ = self.forward(inputs1)
                    z2, q2, _ = self.forward(inputs2)
                    loss = cl_p * self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()


            if ml_num_batch > 0 and cl_num_batch > 0:
                print("Pairwise Total:", round(float(ml_loss.cpu()), 2) + float(cl_loss.cpu()), "ML loss",
                      float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
