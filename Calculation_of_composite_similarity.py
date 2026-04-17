import pandas as pd
import numpy as np

A1 = np.loadtxt("HMDAD/disease-microbe matrix.txt")


def calculate_semantic_similarity(A):

    D = np.where(A > 0, 1, 0)
    num_diseases = D.shape[1]
    DS = np.zeros((num_diseases, num_diseases))
    for i in range(num_diseases):
        D_i = np.where(D[:, i] > 0)[0]
        for j in range(num_diseases):
            D_j = np.where(D[:, j] > 0)[0]
            common = np.intersect1d(D_i, D_j)
            H_di = np.zeros(len(common))
            H_dj = np.zeros(len(common))
            for k in range(len(common)):
                H_di[k] = 1 if common[k] == i else 0.5
                H_dj[k] = 1 if common[k] == j else 0.5
            DS[i, j] = np.sum(H_di + H_dj) / (np.sum(D[:, i]) + np.sum(D[:, j]))
    return DS

def calculate_functional_similarity(A, DS, axis):


     # 计算微生物功能相似性
        num_microbes = A.shape[0]
        FM = np.zeros((num_microbes, num_microbes))
        for i in range(num_microbes):
            d_i3 = np.where(A[i, :] > 0)[0]
            for j in range(num_microbes):
                d_j4 = np.where(A[j, :] > 0)[0]
                max_DE = np.zeros((len(d_i3), len(d_j4)))
                for k in range(len(d_i3)):
                    for l in range(len(d_j4)):
                        max_DE[k, l] = DS[d_i3[k], d_j4[l]]
                FM[i, j] = (np.max(max_DE, axis=1).sum() + np.max(max_DE, axis=0).sum()) / (len(d_i3) + len(d_j4))
        return FM





DS1 = calculate_semantic_similarity(A1)
DS2 = calculate_semantic_similarity(A1.T)



# 计算疾病功能相似性
dfs = calculate_functional_similarity(A1, DS1, axis=1)

# 计算微生物功能相似性
mfs= calculate_functional_similarity(A1.T, DS2, axis=0)

np.savetxt("HMDAD/dfs1.txt", dfs)
np.savetxt("HMDAD/mfs1.txt", mfs)