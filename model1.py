import torch
from torch import nn
from layers import *
import torch.nn.functional as F
import numpy as np
from datadeal import *
import torch.optim as optim
import time
#A = np.loadtxt("MDAD/drug_microbe_matrix.txt")  #MDAD
A = np.loadtxt("Data/disease-microbe matrix.txt") #HMDAD
# A=np.loadtxt("Data1/disease-microbe matrix.txt") #dis
#A = np.loadtxt("aBiofilm/drug_microbe_matrix.txt")    #abiofilm


from layers import CRFAttentionLayer

class GCAN(nn.Module):
    def __init__(self, nfeat, nclass, dropout, alpha, l):
        super(GCAN, self).__init__()
        self.gal = HAGCN(nfeat, nclass, dropout)
        self.crf = CRFAttentionLayer(nclass)  # 加入 CRF-Attention  CRF-Attention层初始化
        self.l = l

    def forward(self, x, adj, sim_mat=None):
        Z1 = self.gal(x, adj)  # GCN嵌入

        if sim_mat is not None:
            sim_mat_tensor = torch.FloatTensor(sim_mat).to(Z1.device)
            Z2 = self.crf(Z1, sim_mat_tensor)  # CRF强化相似性约束  CRF-Attention层forward，Z1是相似性矩阵

        if self.l == 0:
           #MDAD
           #np.savetxt("MDAD/topo embedding1.txt", Z2.detach().cpu().numpy())
            #DIS
         #  np.savetxt("Data1/topo embedding1.txt", Z2.detach().cpu().numpy())
            #HDMAD
            np.savetxt("Data/topo embedding1.txt", Z2.detach().cpu().numpy())
        # elif self.l == 1:
        #     np.savetxt("Data/attr embedding1.txt", Z.detach().cpu().numpy())
        # elif self.l == 2:
        #     np.savetxt("Data/Sdm_dis embedding.txt", Z.detach().cpu().numpy())

        ZZ = torch.sigmoid(torch.matmul(Z1, Z2.T))  # 关联评分
       # return ZZ, Z, Z_crf  # 原始嵌入 Q = Z, 最终嵌入 H = Z_crf
        return ZZ, Z1, Z2  # Z1 是GCN的嵌入，Z2 是CRF-Attention层输出
        #return ZZ

# class GCAN(nn.Module):
#     def __init__(self,nfeat,nclass,dropout,alpha,l): #nfeat 输入的维度331 nclass, 输出的维度 128, alpha 0.2,drouout 0.4
#         super(GCAN, self).__init__()
#         self.gal=HAGCN(nfeat,nclass,dropout)   # 调用HAGCN 的初始化函数
#         self.l=l
#     def forward(self,x,adj):
#         Z=self.gal(x,adj)                      # 调用 HAGCN 的 forward 函数
#         a=Z.detach().numpy() #(331,128)
#         if self.l==0:
#             np.savetxt("Data/topo embedding1.txt",a)
#         if self.l==1:
#             np.savetxt("Data/attr embedding1.txt",a)
#         if self.l == 2:
#             np.savetxt("Data/Sdm_dis embedding.txt", a)
#         # if self.l == 3:
#         #     np.savetxt("./Sm_dis embedding.txt", a)
#         ZZ=torch.sigmoid(torch.matmul(Z,Z.T)) # 331,331
#         return ZZ

def Net_construct(Sr_m,Sm_r):     #异构网络的构建
    N1=np.hstack((Sr_m,A))
    N2=np.hstack((A.T,Sm_r))
    Net=np.vstack((N1,N2))      #(1373+173)*(1373+173)
    return Net
def train33(Net,interaction,l, sim_mat=None):
#def train33(Net,interaction,l):
    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
      # output = model(x, adj)    # 调用GCAN 的forward函数
    ## ***
      #   output = model(x, adj, sim_mat=sim_mat)
      #   loss_train = F.mse_loss(output[idx_train,:], x[idx_train,:])
    ##  ****
        output, Q, H = model(x, adj, sim_mat=sim_mat)
        loss_pred = F.mse_loss(output[idx_train, :], x[idx_train, :])
        loss_crf = crf_loss(Q, H, sim_mat)
        loss_train= loss_pred + 0.9* loss_crf  # 你可以调节系数

        loss_train.backward()
        optimizer.step()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.5f}'.format(loss_train.item()),
              'time: {:.4f}s'.format(time.time() - t))
        return loss_train

    def test():
        model.eval()
    #    output = model(x, adj)
    #     output = model(x, adj, sim_mat=sim_mat)
    #     loss_test = F.mse_loss(output[idx_test,:], x[idx_test,:])
        output, Q, H = model(x, adj, sim_mat=sim_mat)
        loss_pred = F.mse_loss(output[idx_test, :], x[idx_test, :])
        loss_crf = crf_loss(Q, H, sim_mat)
        loss_test = loss_pred + 0.9 * loss_crf  # 你可以调节系数

        print("Test set results:",
              "loss= {:.5f}".format(loss_test.item()))
    adj=torch.FloatTensor(interaction)  # (39,292)
    adj.requires_grad=True  # 设置adj 邻接矩阵参与梯度计算
    x=torch.FloatTensor(Net)   # (331,331)
    x. requires_grad=True   # 设置 x 参与梯度计算
    idx_train = range(240)
    idx_test = range(240, 300)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    model=GCAN(x.shape[1],128,0.4,0.2,l)  # x.shape[1] 第二维的大小 在这里就是列数 331
    # 调用 GCAN 的初始化函数
   #optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)


    t_total = time.time()
    # 200->20
    for epoch in range(25):
        time1 = time.time()
        loss = train(epoch)
        # if loss < 0.1:
        #     break

    print("Optimization Finished!")
    print("Total time elapsed: {:.5f}s".format(time.time() - t_total))

    test()

#
# import torch
# from torch import nn
# from layers import *
# import torch.nn.functional as F
# import numpy as np
# from datadeal import *
# import torch.optim as optim
# import time
# #A = np.loadtxt("MDAD/drug_microbe_matrix.txt")  #MDAD
# A = np.loadtxt("Data/disease-microbe matrix.txt") #HMDAD
# # A=np.loadtxt("Data1/disease-microbe matrix.txt") #dis
# #A = np.loadtxt("aBiofilm/drug_microbe_matrix.txt")    #abiofilm
#
#
# from layers import CRFAttentionLayer
#
# class GCAN(nn.Module):
#     def __init__(self, nfeat, nclass, dropout, l, alpha, beta):
#         super(GCAN, self).__init__()
#         self.gal = HAGCN(nfeat, nclass, dropout)
#         self.crf = CRFAttentionLayer(nclass, alpha, beta)  # 加入 CRF-Attention  CRF-Attention层初始化
#         self.l = l
#
#     def forward(self, x, adj, sim_mat, alpha=None, beta=None):
#         Z1 = self.gal(x, adj)  # GCN嵌入
#
#         if sim_mat is not None:
#             sim_mat_tensor = torch.FloatTensor(sim_mat).to(Z1.device)
#             Z2 = self.crf(Z1, sim_mat_tensor)  # CRF强化相似性约束  CRF-Attention层forward，Z1是相似性矩阵
#
#         if self.l == 0:
#            #MDAD
#            #np.savetxt("MDAD/topo embedding1.txt", Z2.detach().cpu().numpy())
#             #DIS
#          #  np.savetxt("Data1/topo embedding1.txt", Z2.detach().cpu().numpy())
#             #HDMAD
#             np.savetxt("Data/topo embedding1.txt", Z2.detach().cpu().numpy())
#         # elif self.l == 1:
#         #     np.savetxt("Data/attr embedding1.txt", Z.detach().cpu().numpy())
#         # elif self.l == 2:
#         #     np.savetxt("Data/Sdm_dis embedding.txt", Z.detach().cpu().numpy())
#
#         ZZ = torch.sigmoid(torch.matmul(Z1, Z2.T))  # 关联评分
#        # return ZZ, Z, Z_crf  # 原始嵌入 Q = Z, 最终嵌入 H = Z_crf
#         return ZZ, Z1, Z2  # Z1 是GCN的嵌入，Z2 是CRF-Attention层输出
#         #return ZZ
#
# #
# #
# # class GCAN(nn.Module):
# #     def __init__(self,nfeat,nclass,dropout,alpha,l): #nfeat 输入的维度331 nclass, 输出的维度 128, alpha 0.2,drouout 0.4
# #         super(GCAN, self).__init__()
# #         self.gal=HAGCN(nfeat,nclass,dropout)   # 调用HAGCN 的初始化函数
# #         self.l=l
# #     def forward(self,x,adj):
# #         Z=self.gal(x,adj)                      # 调用 HAGCN 的 forward 函数
# #         a=Z.detach().numpy() #(331,128)
# #         if self.l==0:
# #             np.savetxt("Data/topo embedding1.txt",a)
# #         if self.l==1:
# #             np.savetxt("Data/attr embedding1.txt",a)
# #         if self.l == 2:
# #             np.savetxt("Data/Sdm_dis embedding.txt", a)
# #         # if self.l == 3:
# #         #     np.savetxt("./Sm_dis embedding.txt", a)
# #         ZZ=torch.sigmoid(torch.matmul(Z,Z.T)) # 331,331
# #         return ZZ
#
# def Net_construct(Sr_m,Sm_r):     #异构网络的构建
#     N1=np.hstack((Sr_m,A))
#     N2=np.hstack((A.T,Sm_r))
#     Net=np.vstack((N1,N2))      #(1373+173)*(1373+173)
#     return Net
# def train33(Net,interaction,l, sim_mat=None, alpha=None, beta=None):
# #def train33(Net,interaction,l):
#     def train(epoch):
#         t = time.time()
#         model.train()
#         optimizer.zero_grad()
#       # output = model(x, adj)    # 调用GCAN 的forward函数
#     ## ***
#       #   output = model(x, adj, sim_mat=sim_mat)
#       #   loss_train = F.mse_loss(output[idx_train,:], x[idx_train,:])
#     ##  ****
#         output, Q, H = model(x, adj, sim_mat=sim_mat, alpha=alpha, beta=beta)
#         loss_pred = F.mse_loss(output[idx_train, :], x[idx_train, :])
#         loss_crf = crf_loss(Q, H, sim_mat)
#         loss_train= loss_pred + 0.5* loss_crf  # 你可以调节系数
#
#         loss_train.backward()
#         optimizer.step()
#         print('Epoch: {:04d}'.format(epoch + 1),
#               'loss_train: {:.5f}'.format(loss_train.item()),
#               'time: {:.4f}s'.format(time.time() - t))
#         return loss_train
#
#     def test():
#         model.eval()
#     #    output = model(x, adj)
#     #     output = model(x, adj, sim_mat=sim_mat)
#     #     loss_test = F.mse_loss(output[idx_test,:], x[idx_test,:])
#         output, Q, H = model(x, adj, sim_mat=sim_mat)
#         loss_pred = F.mse_loss(output[idx_test, :], x[idx_test, :])
#         loss_crf = crf_loss(Q, H, sim_mat)
#         loss_test = loss_pred + 0.1 * loss_crf  # 你可以调节系数
#
#         print("Test set results:",
#               "loss= {:.5f}".format(loss_test.item()))
#     adj=torch.FloatTensor(interaction)  # (39,292)
#     adj.requires_grad=True  # 设置adj 邻接矩阵参与梯度计算
#     x=torch.FloatTensor(Net)   # (331,331)
#     x. requires_grad=True   # 设置 x 参与梯度计算
#     idx_train = range(240)
#     idx_test = range(240, 300)
#     idx_train = torch.LongTensor(idx_train)
#     idx_test = torch.LongTensor(idx_test)
#     model=GCAN(x.shape[1],128,0.4,0.2,alpha,beta)  # x.shape[1] 第二维的大小 在这里就是列数 331
#     # 调用 GCAN 的初始化函数
#     optimizer = optim.Adam(model.parameters(),lr=0.01, weight_decay=5e-4)
#
#
#     t_total = time.time()
#     # 200->20
#     for epoch in range(25):
#         loss = train(epoch)
#         # if loss < 0.1:
#         #     break
#
#     print("Optimization Finished!")
#     print("Total time elapsed: {:.5f}s".format(time.time() - t_total))
#
#     test()
#











#
# class EarlyStopping:
#     def __init__(self, patience=5, delta=1e-4):
#         self.patience = patience  # 容忍轮数
#         self.delta = delta        # 最小变化幅度
#         self.best_loss = None
#         self.counter = 0
#         self.early_stop = False
#
#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss - self.delta:
#             self.counter += 1
#             print(f"早停计数：{self.counter}/{self.patience}")
#             if self.counter >= self.patience:
#                 print(">>> 提前停止训练")
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0
#
#
# def train33(Net, interaction, l, sim_mat=None):
#     def train(epoch):
#         t = time.time()
#         model.train()
#         optimizer.zero_grad()
#         output, Q, H = model(x, adj, sim_mat=sim_mat)
#         loss_pred = F.mse_loss(output[idx_train, :], x[idx_train, :])
#         loss_crf = crf_loss(Q, H, sim_mat)
#         loss_train = loss_pred + 0.5 * loss_crf
#         loss_train.backward()
#         optimizer.step()
#         print('Epoch: {:04d} | Train Loss: {:.5f} | Time: {:.2f}s'.format(
#             epoch + 1, loss_train.item(), time.time() - t))
#         return loss_train.item()
#
#     def test():
#         model.eval()
#         with torch.no_grad():
#             output, Q, H = model(x, adj, sim_mat=sim_mat)
#             loss_pred = F.mse_loss(output[idx_test, :], x[idx_test, :])
#             loss_crf = crf_loss(Q, H, sim_mat)
#             loss_test = loss_pred + 0.5 * loss_crf
#         print("Test set results: loss = {:.5f}".format(loss_test.item()))
#         return loss_test.item()
#
#     # 数据准备
#     adj = torch.FloatTensor(interaction)
#     adj.requires_grad = True
#     x = torch.FloatTensor(Net)
#     x.requires_grad = True
#
#     idx_train = torch.LongTensor(range(240))
#     idx_test = torch.LongTensor(range(240, 300))
#
#     model = GCAN(x.shape[1], 128, 0.4, 0.2, l)
#     optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
#     early_stopping = EarlyStopping(patience=5, delta=1e-4)
#
#     t_total = time.time()
#     max_epochs = 100
#
#     for epoch in range(max_epochs):
#         train_loss = train(epoch)
#         val_loss = test()
#
#         early_stopping(val_loss)
#         if early_stopping.early_stop:
#             break
#
#     print("Optimization Finished!")
#     print("Total time elapsed: {:.5f}s".format(time.time() - t_total))
#
#     final_loss = test()
#     print("Final Test Loss: {:.5f}".format(final_loss))
#
