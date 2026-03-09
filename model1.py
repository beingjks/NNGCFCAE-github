import torch
from torch import nn
from layers import *
import torch.nn.functional as F
import numpy as np
from datadeal import *
import torch.optim as optim
import time

A = np.loadtxt("HMDAD/disease-microbe matrix.txt")

from layers import CRFAttentionLayer


class GCAN(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha, l):
        super(GCAN, self).__init__()
        self.gal = HAGCN(nfeat, nclass, dropout)
        self.crf = CRFAttentionLayer(nclass)
        self.l = l

    def forward(self, x, adj, sim_mat=None):
        Z1 = self.gal(x, adj)

        if sim_mat is not None:
            sim_mat_tensor = torch.FloatTensor(sim_mat).to(Z1.device)
            Z2 = self.crf(Z1, sim_mat_tensor)

        if self.l == 0:
            np.savetxt("HMDAD/topo embedding1.txt", Z2.detach().cpu().numpy())

        ZZ = torch.sigmoid(torch.matmul(Z1, Z2.T))
        return ZZ, Z1, Z2


def Net_construct(Sr_m, Sm_r):
    N1 = np.hstack((Sr_m, A))
    N2 = np.hstack((A.T, Sm_r))
    Net = np.vstack((N1, N2))
    return Net


def train33(Net, interaction, l, sim_mat=None):
    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output, Q, H = model(x, adj, sim_mat=sim_mat)
        loss_pred = F.mse_loss(output[idx_train, :], x[idx_train, :])
        loss_crf = crf_loss(Q, H, sim_mat)
        loss_train = loss_pred + 0.9 * loss_crf
        loss_train.backward()
        optimizer.step()
        print(
            'Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.5f}'.format(loss_train.item()),
            'time: {:.4f}s'.format(time.time() - t)
        )
        return loss_train

    def test():
        model.eval()
        output, Q, H = model(x, adj, sim_mat=sim_mat)
        loss_pred = F.mse_loss(output[idx_test, :], x[idx_test, :])
        loss_crf = crf_loss(Q, H, sim_mat)
        loss_test = loss_pred + 0.9 * loss_crf
        print("Test set results:", "loss= {:.5f}".format(loss_test.item()))

    adj = torch.FloatTensor(interaction)
    adj.requires_grad = True
    x = torch.FloatTensor(Net)
    x.requires_grad = True
    idx_train = range(240)
    idx_test = range(240, 300)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    model = GCAN(x.shape[1], 128, 0.4, 0.2, l)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    t_total = time.time()
    for epoch in range(25):
        loss = train(epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.5f}s".format(time.time() - t_total))

    test()
