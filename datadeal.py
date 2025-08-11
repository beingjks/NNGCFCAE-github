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


#版本1
# def NNM2():
#     # === 加载微生物-药物关联矩阵 A ===
#     A = np.loadtxt("Data/disease-microbe matrix.txt")  # 原始观测矩阵
#     m, n = A.shape
#     mask = (A != 0).astype(float)  # 仅观测位置的掩码
#    # H = A.copy()  # H 即公式中的已知矩阵
#     H= np.loadtxt("net.txt")
#     # === 参数设置 ===
#     omega = 10  # 正则项权重
#     gamma = 1.0  # ADMM 步长
#     alpha = 0.75  # 用于公式23融合BAN结果（可选）
#     max_iter = 200
#     tol = 1e-4
#
#     # === 初始化变量 ===
#     E = np.zeros((m, n))  # 预测矩阵
#     Y = np.zeros((m, n))  # 辅助变量
#     Z = np.zeros((m, n))  # 拉格朗日乘子
#
#     # === Step 1: ADMM迭代 ===
#     for k in range(max_iter):
#         # --- 更新 E（Eq. 20） ---
#         temp = Y - Z / gamma
#         right = mask * H + (gamma * temp + omega * H) / (gamma + omega)
#         E_new = singular_value_thresholding(right, 1 / (gamma + omega))
#
#         # --- 更新 Y（Eq. 21） ---
#         temp2 = E_new + Z / gamma
#         U, S, Vt = np.linalg.svd(temp2, full_matrices=False)
#         S_thresh = np.maximum(S - 1 / gamma, 0)
#         Y_new = U @ np.diag(S_thresh) @ Vt
#
#         # --- 更新 Z（Eq. 22） ---
#         Z_new = Z + gamma * (E_new - Y_new)
#
#         # --- 检查收敛 ---
#         err = np.linalg.norm(E_new - E, ord="fro") / np.linalg.norm(E, ord="fro") if np.linalg.norm(E) > 0 else 1.0
#         E, Y, Z = E_new, Y_new, Z_new
#
#         if err < tol:
#             print(f"✅ ADMM 收敛：第 {k} 次迭代，相对误差 {err:.5e}")
#             break
#
#     # === Step 2: 融合BAN结果（Eq. 23）===
#     # 如果没有BAN模块 M1，可以先用全0代替
#     try:
#         M1 = np.loadtxt("ban_prediction.txt")  # BAN输出的预测矩阵
#     except:
#         M1 = np.zeros_like(E)
#         print("⚠️ 未提供BAN输出结果，使用全零矩阵")
#
#     # M_final = alpha * M1 + (1 - alpha) * E
#
#     # === 保存最终预测结果 ===
#     print("✅ 最终预测结果（仅 NNM）已保存。")
#     np.savetxt("predicted_microbe_drug_by_NNM.txt", E, fmt="%.4f")
#     # np.savetxt("final_microbe_drug_prediction.txt", M_final, fmt="%.4f")
#
#     # print("✅ 最终预测结果（融合 BAN + NNM）已保存。")
#     return E

def singular_value_thresholding(X, tau):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S_thresh = np.maximum(S - tau, 0)
    return U @ np.diag(S_thresh) @ Vt



# def crf_loss(Q, H, sim_mat, alpha=50, beta=1):    # 计算CRF损失函数
#
#     """
#     Q: 原始节点表示 (N, d)，来自 GCN
#     H: CRF 层输出的节点表示 (N, d)
#     sim_mat: 节点相似性矩阵 (N, N)，通常为疾病/微生物相似性
#     """
#     # 自一致性损失
#     loss_self = alpha * F.mse_loss(H, Q)
#
#     # 邻居相似性损失
#     diff = H.unsqueeze(1) - H.unsqueeze(0)  # (N, N, d)
#     sq_diff = torch.sum(diff ** 2, dim=2)   # (N, N)
#     if isinstance(sim_mat, np.ndarray):
#         sim_mat = torch.FloatTensor(sim_mat)
#     sim_mat = sim_mat.to(H.device)
#
#     sim_mat = sim_mat.to(H.device)
#
#     loss_pair = beta * torch.sum(sim_mat * sq_diff)
#
#     return loss_self + loss_pair
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

# # 版本2
# import numpy as np
# import cvxpy as cp
#
#
# def nuclear_norm_minimization(omega=10, gamma=1, max_iter=100, tol=1e-5):
#     """
#     A: 原始微生物-药物关联矩阵 (部分观测)
#     H: 综合相似性矩阵 (作为参考)
#     omega: 正则项系数
#     gamma: ADMM 惩罚参数
#     max_iter: 最大迭代次数
#     tol: 收敛阈值
#     """
#     A= np.loadtxt("Data/disease-microbe matrix.txt")
#     H = np.loadtxt("net.txt")
#     m, n = A.shape
#     # 初始化变量
#     E = np.zeros((m, n))
#     Y = np.zeros((m, n))
#     Z = np.zeros((m, n))
#
#     # 定义观测索引集 Ω
#     Omega = (A != 0)
#
#     for k in range(max_iter):
#         # Step 1: 更新 E (使用奇异值软阈值化)
#         V = Y - Z
#         V[Omega] = (H[Omega] + gamma * V[Omega]) / (1 + gamma)
#         U, s, Vt = np.linalg.svd(V, full_matrices=False)
#         s_threshold = np.maximum(s - omega / gamma, 0)
#         E_new = (U * s_threshold) @ Vt
#
#         # Step 2: 更新 Y
#         Y_new = (gamma * E_new + Z) / (1 + gamma)
#
#         # Step 3: 更新 Z
#         Z_new = Z + E_new - Y_new
#
#         # 检查收敛
#         if np.linalg.norm(E_new - E, 'fro') < tol:
#             break
#
#         E, Y, Z = E_new, Y_new, Z_new
#
#     return E


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
    # omega =10  # 在那个HDMAD好
    # gamma = 1
    omega = 1.0  # 调低阈值
    gamma = 1
    # omega = 0.5
    # gamma = 5
    # omega = 0.2
    # gamma = 2
    """
    仅保留非常强的结构 omega = 10, gamma = 1 （你当前配置）
    保留更多真实信息（推荐） omega = 1, gamma = 1 或 omega = 2
    极度保守、保留更多尾部信号  omega = 0.1  ~ 0.5，gamma = 1
    """


    max_iter = 100
    tol = 1e-5
   # H =np.loadtxt("net.txt")
    E = np.zeros_like(H)
    Y = np.zeros_like(H)
    Z = np.zeros_like(H)

    for _ in range(max_iter):
        V = Y - Z
     #####改后   V[Omega] = H[Omega]  # 直接用观测值覆盖

        V[Omega] = (H[Omega] + gamma * V[Omega]) / (1 + gamma)
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
        # 改后
        # U0, s0, Vt0 = np.linalg.svd(H, full_matrices=False)
        # print("Top 5 singular values:", s0[:5])
        #原
        s_threshold = np.maximum(s - omega / gamma, 0)
        E_new = (U * s_threshold) @ Vt
        # 改后
        # s_threshold = np.maximum(s - omega / gamma, 0)
        # E_new = np.dot(U, np.dot(np.diag(s_threshold), Vt))
        # E_new[E_new < 0] = 0  # 强制非负

        Y_new = (gamma * E_new + Z) / (1 + gamma)
        Z_new = Z + E_new - Y_new

        if np.linalg.norm(E_new - E, 'fro') < tol:
            break

        E, Y, Z = E_new, Y_new, Z_new

    return E


def extract_predicted_A(E, A_shape):
    m, n = A_shape
    return E[:m, m:]  # 提取右上角部分作为 A_pred



# def nuclear_norm_minimization( H,Omega,omega,gamma):
#     # omega =10  # 在那个HDMAD好
#     # gamma = 1
#     # omega = 1.0  # 调低阈值
#     # gamma = 1
#
#     max_iter = 100
#     tol = 1e-5
#    # H =np.loadtxt("net.txt")
#     E = np.zeros_like(H)
#     Y = np.zeros_like(H)
#     Z = np.zeros_like(H)
#
#     for _ in range(max_iter):
#         V = Y - Z
#         V[Omega] = (H[Omega] + gamma * V[Omega]) / (1 + gamma)
#         U, s, Vt = np.linalg.svd(V, full_matrices=False)
#         s_threshold = np.maximum(s - omega / gamma, 0)
#         E_new = (U * s_threshold) @ Vt
#
#         Y_new = (gamma * E_new + Z) / (1 + gamma)
#         Z_new = Z + E_new - Y_new
#
#         if np.linalg.norm(E_new - E, 'fro') < tol:
#             break
#
#         E, Y, Z = E_new, Y_new, Z_new
#
#     return E