import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from CSEAE import *
from Feature_matrix import *
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import preprocessing
from model1 import *
import pandas as pd

A = np.loadtxt("HMDAD/disease-microbe matrix.txt")
Sd_che = np.loadtxt("HMDAD/dfs.txt")
Sm_f = np.loadtxt("HMDAD/mfs.txt")
Sd_dis = np.loadtxt("HMDAD/d-dis.txt")
Sm_dis = np.loadtxt("HMDAD/m-dis.txt")
known = np.loadtxt("HMDAD/known1.txt")
unknown = np.loadtxt("HMDAD/unknown1.txt")

TP, TN, FP, FN = 0, 0, 0, 0
scores = []
tlabels = []


def kflod_5(num):
    k = []
    unk = []
    lk = len(known)
    luk = len(unknown)
    for i in range(lk):
        k.append(i)
    for i in range(luk):
        unk.append(i)
    random.shuffle(k)
    random.shuffle(unk)

    ratio = 1

    for cv in range(1, 6):
        interaction = np.array(list(A))

        if cv < 5:
            B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]
            B2_raw = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]
        else:
            B1 = known[k[(cv - 1) * (lk // 5):lk], :]
            B2_raw = unknown[unk[(cv - 1) * (luk // 5):luk], :]

        num_pos = B1.shape[0]
        num_neg_wanted = ratio * num_pos
        if B2_raw.shape[0] > num_neg_wanted:
            idx = np.random.choice(B2_raw.shape[0], num_neg_wanted, replace=False)
            B2 = B2_raw[idx]
        else:
            B2 = B2_raw

        for i in range(B1.shape[0]):
            interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0

        Sr_m_HIP = HIP_Calculate(interaction)
        Sm_r_HIP = HIP_Calculate(interaction.T)
        Sm_r_GIP = GIP_Calculate(interaction)
        Sr_m_GIP = GIP_Calculate1(interaction)

        Sr_m = (Sr_m_HIP + Sr_m_GIP) / 2
        Sm_r = (Sm_r_HIP + Sm_r_GIP) / 2

        Net = Net_construct(Sr_m, Sm_r)

        Y = Sr_m
        W = Sm_r
        H = construct_H(A, Y, W)
        Omega = get_observed_mask(A)
        E = nuclear_norm_minimization(H, Omega)
        A1 = extract_predicted_A(E, A.shape)
        # np.savetxt("A1.txt", A1)

        train33(Net, interaction, 0, sim_mat=Net)

        Sdd = RWR(Sr_m)
        Smm = RWR(Sm_r)

        np.savetxt("HMDAD/Sdd.txt", Sdd)
        np.savetxt("HMDAD/Smm.txt", Smm)

        A_r = np.hstack((np.hstack((np.hstack((interaction, Sd_che)), interaction)), Sdd))
        A_m = np.hstack((np.hstack((np.hstack((interaction.T, Sm_f)), interaction.T)), Smm))

        A_r = np.hstack((A_r, Sd_dis))
        A_m = np.hstack((A_m, Sm_dis))

        train2(A_r, 0)
        train2(A_m, 1)

        df, mf = Fmatrix(interaction)
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)
        mf = min_max_scaler.fit_transform(mf)

        score = torch.sigmoid(torch.FloatTensor(np.dot(df, mf.T)))
      #  df_out = pd.DataFrame(score)
     #   df_out.to_excel('score.xlsx', index=False, header=False)
        score = np.array(score)

        for i in range(len(B1)):
            index1 = int(B1[i, 0] - 1)
            index2 = int(B1[i, 1] - 1)
            scores.append(score[index1, index2])
            tlabels.append(A[index1, index2])
        for i in range(len(B2)):
            index1 = int(B2[i, 0] - 1)
            index2 = int(B2[i, 1] - 1)
            scores.append(score[index1, index2])
            tlabels.append(A[index1, index2])

        print("fold cv--{}".format(cv))

    fpr, tpr, threshold = roc_curve(tlabels, scores)
    num = str(num)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/fpr" + num + ".txt", fpr)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/tpr" + num + ".txt", tpr)
    np.savetxt("score GCAT=0.05SAE=0.01/score" + num + ".txt", score)
    np.savetxt("tlable GCAT=0.05SAE=0.01/tlable" + num + ".txt", tlabels)

    auc_val = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(tlabels, scores)
    aupr_score = auc(recall, precision)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/precision" + num + ".txt", precision)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/recall" + num + ".txt", recall)

    print("auc:", auc_val)
    print("aupr:", aupr_score)
    return auc_val, aupr_score


auc_val = []
aupr_val = []
for i in range(10):
    auc1, aupr = kflod_5(i)
    print("------------------------------")
    auc_val.append(auc1)
    aupr_val.append(aupr)
    np.savetxt("AUC--------CSAEtt/auc.txt", auc_val)
    np.savetxt("AUC--------CSAEtt/aupr.txt", aupr_val)

print("auc_val:", sum(auc_val) / 10)
print("aupr_val:", sum(aupr_val) / 10)
print("----------------------------------------------------------------------------------------------------------------")
print("训练完毕")
