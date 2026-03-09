import numpy as np
import math
import  torch
import torch.nn.functional as F
# def Net_construct(Sr_m,Sm_r):     #异构网络的构建
#     N1=np.hstack((Sr_m,A))
#     N2=np.hstack((A.T,Sm_r))
#     Net=np.vstack((N1,N2))      #(1373+173)*(1373+173)
#     return Net                  # HN



def HIP_Calculate(M):
    l=len(M)
    cl=np.size(M,axis=1)
    SM=np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            dnum = 0
            for k in range(cl):
                if M[i][k]!=M[j][k]:
                    dnum=dnum+1
            SM[i][j]=1-dnum/cl             #HIP计算出来的相似矩阵
    return SM

def GIP_Calculate(M):     #计算高斯核相似性
    l=np.size(M,axis=1)
    sm=[]
    m=np.zeros((l,l))
    for i in range(l):
        tmp=(np.linalg.norm(M[:,i]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[:,i]-M[:,j]))**2))
    return m
def GIP_Calculate1(M):     #计算高斯核相似性
    l=np.size(M,axis=0)
    sm=[]
    m=np.zeros((l,l))
    km=np.zeros((l,l))
    for i in range(l):
        tmp=(np.linalg.norm(M[i,:]))**2
        sm.append(tmp)
    gama=l/np.sum(sm)
    for i in range(l):
        for j in range(l):
            m[i,j]=np.exp(-gama*((np.linalg.norm(M[i,:]-M[j,:]))**2))
    for i in range(l):
        for j in range(l):
            km[i,j]=1/(1+np.exp(-15*m[i,j]+math.log(9999)))
    return km
def Cosine_Sim(M):
    l=len(M)
    SM = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            v1=np.dot(M[i],M[j])
            v2=np.linalg.norm(M[i],ord=2)
            v3=np.linalg.norm(M[j],ord=2)
            if v2*v3==0:
                SM[i][j]=0
            else:
                SM[i][j]=v1/(v2*v3)
    return SM

def RWR(SM):
    alpha = 0.1
    E = np.identity(len(SM))  # 单位矩阵
    M = np.zeros((len(SM), len(SM)))
    s=[]
    for i in range(len(M)):
        for j in range(len(M)):
            M[i][j] = SM[i][j] / (np.sum(SM[i, :]))
    for i in range(len(M)):
        e_i = E[i, :]
        p_i1 = np.copy(e_i)
        for j in range(10):
            p_i = np.copy(p_i1)
            p_i1 = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        s.append(p_i1)
    return s



def singular_value_thresholding(X, tau):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0)
    return U @ np.diag(S_thresh) @ Vt



def crf_loss(Q, H, sim_mat, alpha=50, beta=1):    # 计算CRF损失函数

    """
    Q: 原始节点表示 (N, d)，来自 GCN
    H: CRF 层输出的节点表示 (N, d)
    sim_mat: 节点相似性矩阵 (N, N)，通常为疾病/微生物相似性
    """
    # 自一致性损失
    loss_self = alpha * F.mse_loss(H, Q)

    # 邻居相似性损失
    diff = H.unsqueeze(1) - H.unsqueeze(0)  # (N, N, d)
    sq_diff = torch.sum(diff ** 2, dim=2)   # (N, N)
    if isinstance(sim_mat, np.ndarray):
        sim_mat = torch.FloatTensor(sim_mat)
    sim_mat = sim_mat.to(H.device)

    sim_mat = sim_mat.to(H.device)

    loss_pair = beta * torch.sum(sim_mat * sq_diff)

    return loss_self + loss_pair


def construct_H(A, Y, W):
    """
    构建 H 矩阵：H = [[Y, A], [A^T, W]]
    A: (39, 292)
    Y: (39, 39)
    W: (292, 292)
    返回 H: (331, 331)
    """
    m, n = A.shape  # m=39, n=292
    H = np.zeros((m + n, m + n))
    H[:m, :m] = Y         # 左上角：微生物相似性
    H[m:, m:] = W         # 右下角：药物相似性
    H[:m, m:] = A         # 右上角：A
    H[m:, :m] = A.T       # 左下角：A^T
    return H
def get_observed_mask(A):
    """
    生成 Ω 掩码：只标记 A 部分的位置为 True
    """
    m, n = A.shape
    Omega = np.zeros((m + n, m + n), dtype=bool)
    Omega[:m, m:] = A != 0      # A 区域（右上角）
    Omega[m:, :m] = A.T != 0    # A^T 区域（左下角）
    return Omega

def nuclear_norm_minimization( H,Omega):

    omega = 1.0  # 调低阈值
    gamma = 1


    max_iter = 100
    tol = 1e-5
   # H =np.loadtxt("net.txt")
    E = np.zeros_like(H)
    Y = np.zeros_like(H)
    Z = np.zeros_like(H)

    for _ in range(max_iter):
        V = Y - Z

        V[Omega] = (H[Omega] + gamma * V[Omega]) / (1 + gamma)
        U, s, Vt = np.linalg.svd(V, full_matrices=False)

        #原
        s_threshold = np.maximum(s - omega / gamma, 0)
        E_new = (U * s_threshold) @ Vt


        Y_new = (gamma * E_new + Z) / (1 + gamma)
        Z_new = Z + E_new - Y_new

        if np.linalg.norm(E_new - E, 'fro') < tol:
            break

        E, Y, Z = E_new, Y_new, Z_new

    return E


def extract_predicted_A(E, A_shape):
    m, n = A_shape
    return E[:m, m:]  # 提取右上角部分作为 A_pred


