import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.nn.functional as F
from torch import nn

class CRFAttentionLayer(nn.Module):      #  CRF attention层
    def __init__(self, in_dim, alpha=50, beta=1, K=2):
        super(CRFAttentionLayer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.att_proj = nn.Linear(in_dim, in_dim)

    def forward(self, Q, sim_mat):
        """
        Q: 原始节点表示 (N, d)
        sim_mat: 相似性矩阵 (N, N)，如疾病/微生物的结构相似性
        """
        H = Q
        sim_mat = sim_mat.to(Q.device)

        for _ in range(self.K):
            H_proj = self.att_proj(H)
            att_scores = torch.matmul(H_proj, H_proj.T)
            att_scores = att_scores.masked_fill(sim_mat == 0, float('-inf'))  # 屏蔽无连接
            lamb = torch.softmax(att_scores, dim=1)  # attention 权重

            numerator = self.alpha * Q + self.beta * torch.matmul(lamb, H)
            denominator = self.alpha + self.beta * lamb.sum(dim=1, keepdim=True)
            H = numerator / denominator
        return H










def glorot_init(input_dim, output_dim):
   init_range = np.sqrt(6.0/(input_dim + output_dim))
   initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
   return nn.Parameter(initial)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight=nn.Parameter(torch.Tensor(in_features,out_features))
        #self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        #self.weight = torch.empty(in_features, out_features, requires_grad=True)
        #self.weight=glorot_init(in_features,out_features)
        #self.weight=torch.empty(in_features,out_features)
        #self.weight=glorot_init(in_features,out_features)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.num=num
        self.reset_parameters1(num)
        #self.reset_parameters(num)

    def reset_parameters2(self, num):
        self.weight = torch.nn.init.uniform_(self.weight, a=0.0, b=1.0)
        numm = str(num)
        np.savetxt("HMDAD/weight" + numm + ".txt", self.weight.detach().numpy())
    def reset_parameters1(self,num):
        self.weight=torch.nn.init.kaiming_normal_(self.weight,mode='fan_in',nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        numm = str(num)
        np.savetxt("HMDAD/weight"+numm+".txt",self.weight.detach().numpy())
    def reset_parameters(self,num):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        numm=str(num)
        np.savetxt("HMDAD/weight"+numm+".txt",self.weight.detach().numpy())

    def forward(self, input, adj):
        #self.weight = torch.nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        #self.weight=torch.nn.init.xavier_normal_(self.weight,gain=nn.init.calculate_gain('relu'))
        #self.reset_parameters1(self.num)
        #self.reset_parameters(self.num)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output+=self.bias.to(device)
        #
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvSparse(nn.Module):
   def __init__(self, input_dim, output_dim, adj, activation = F.relu):
      super(GraphConvSparse, self).__init__()
      self.weight = glorot_init(input_dim, output_dim)
      self.adj = adj
      self.activation = activation

   def forward(self, inputs):
      x = inputs
      x = torch.mm(x,self.weight)
      x = torch.mm(self.adj, x)
      outputs = self.activation(x)
      return outputs


class GraphAttention(Module):
    def __init__(self,in_features,out_features,dropout,alpha,concat=True):
        super(GraphAttention, self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.dropout=dropout
        self.alpha=alpha
        self.concat=concat

        self.W=nn.Parameter(torch.zeros(size=(in_features,out_features)))
        nn.init.xavier_uniform_(self.W.data,gain=1.414)
        self.a=nn.Parameter(torch.zeros(size=(2*out_features,1)))
        nn.init.xavier_uniform_(self.a.data,gain=1.414)

        self.leakyrelu=nn.LeakyReLU(self.alpha)
    def forward(self,inp,adj):
        h=torch.mm(inp,self.W)
        N=h.size()[0]
        a_input=torch.cat([h.repeat(1,N).view(N*N,-1),h.repeat(N,1)],dim=1).view(N,-1,2*self.out_features)
        e=self.leakyrelu(torch.matmul(a_input,self.a).squeeze(2))

        zero_vec=-1e12*torch.ones_like(e)
        attention=torch.where(adj>0,e,zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime
    def __repr__(self):
        return self.__class__.__name__+' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

def leakyrelu(x):
    return torch.max(0.1 * x, x)





class HAGCN(nn.Module):
    def __init__(self, in_features, out_features, dropout):  # in_features=331, out_features=128, dropout=0.4
        super(HAGCN, self).__init__()
        self.in_features = in_features  # 输入特征的维度
        self.out_features = out_features  # 输出特征的维度
        self.dropout = dropout  # Dropout概率
        self.linear = nn.Linear(in_features, out_features)  # 线性变换层

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # 可学习的权重矩阵 (331, 128)
        nn.init.xavier_normal_(self.W.data, gain=1.414)  # 使用Xavier正态分布初始化权重矩阵
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # 注意力分数矩阵 (256, 1)
        nn.init.xavier_normal_(self.a.data, gain=1.414)  # 使用Xavier正态分布初始化注意力分数矩阵

    def forward(self, x, adj):
        h = self.linear(x)  # 输入x经过线性变换得到h x(331,331)
        N = h.size(0)  # 获取节点数量 #N:331

        # 计算注意力分数
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)  # 331,331,256 # 构建注意力机制的输入
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))  # 计算每个节点的注意力权重  e(331,331)
        zero_vec = -9e15 * torch.ones_like(e)  # 创建一个与e形状相同的全负无穷矩阵 (331,331)
        attention = torch.where(torch.Tensor(x) > 0, e, zero_vec)  # (331,331)根据输入x的值选择注意力权重或负无穷
        attention = F.softmax(attention, dim=1)  # 对注意力权重进行softmax归一化 (331,331)
        A = F.dropout(attention, self.dropout, training=self.training)  #(331,331) 对注意力权重进行Dropout

        degree1 = torch.sum(adj, dim=1)  # 计算邻接矩阵的度（行和）39
        degree2 = torch.sum(adj, dim=0)  # 计算邻接矩阵的度（列和）292
        degree = torch.cat((degree1, degree2), dim=0)  # 将行和和列和拼接在一起 331
        D = torch.diag(degree)  # 构建度矩阵 (331,331)
        IN_D_inv_sqrt = torch.sqrt(torch.inverse(torch.eye(x.size(0)) + D))  # (331,331) 计算度矩阵的逆平方根
        # x.size(0) 是节点特征矩阵 x 的行数，即图中节点的数量
        # torch.eye(x.size(0)) 生成一个大小为  n×n 的单位矩阵，其中 n 是节点数量
        Dd1 = torch.sum(A, dim=1)  # 计算注意力矩阵的度（行和）331
        Dd = torch.diag(Dd1)  # 构建注意力矩阵的度矩阵 (331,331)
        #Tgcn = torch.matmul(torch.matmul((torch.matmul(IN_D_inv_sqrt, Dd)), torch.Tensor(A)).T, IN_D_inv_sqrt)  # 计算图卷积的变换矩阵
        Dd = Dd + torch.eye(Dd.size(0)) * 1e-6  # 对角元素加上一个小值
        Tgcn = torch.matmul(torch.matmul((torch.matmul(IN_D_inv_sqrt, torch.inverse(Dd))), torch.Tensor(A)),IN_D_inv_sqrt)  # 计算图卷积的变换矩阵
        x = torch.mm(Tgcn, x)  # 对输入x进行图卷积操作  (331,331)
        x = torch.mm(x, self.W)  # 对图卷积结果进行线性变换  (331,128)
        x = F.relu(x)  # 对结果应用ReLU激活函数  (331,128)

        return x  # 返回最终的输出






