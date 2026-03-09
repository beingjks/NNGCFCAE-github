import numpy as np
import pandas as pd


def Fmatrix(M):
    A1 = np.loadtxt("A1.txt")
    A = M

    Z = np.loadtxt("HMDAD/topo embedding1.txt")
    Z_d = Z[0:39, :]
    Z_m = Z[39:331, :]
    A_dd = np.loadtxt("HMDAD/attr embedding_d.txt")
    A_mm = np.loadtxt("HMDAD/attr embedding_m.txt")
    Sd_che = np.loadtxt("HMDAD/dfs.txt")
    Sm_fun = np.loadtxt("HMDAD/mfs.txt")
    Sd_dis = np.loadtxt("HMDAD/d-dis.txt")
    Sm_dis = np.loadtxt("HMDAD/m-dis.txt")
    Sdd = np.loadtxt("HMDAD/Sdd.txt")
    Smm = np.loadtxt("HMDAD/Smm.txt")

    Disease_feature = np.hstack((Z_d, np.hstack((A_dd, np.hstack((Sd_che, A))))))
    Disease_feature = np.hstack((np.hstack((Disease_feature, Sd_dis)), A))
    Disease_feature = np.hstack((np.hstack((Disease_feature, Sdd)), A1))

    Microbe_feature = np.hstack((Z_m, np.hstack((A_mm, np.hstack((A.T, Sm_fun))))))
    Microbe_feature = np.hstack((np.hstack((Microbe_feature, A.T)), Sm_dis))
    Microbe_feature = np.hstack((np.hstack((Microbe_feature, A1.T)), Smm))

    return Disease_feature, Microbe_feature
