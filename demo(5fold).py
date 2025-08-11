import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from CSEAE import *

from Feature_matrix import *
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import preprocessing
from model1 import *
# 原代码
# # ########## 已经能复现原来的性能效果了  0.97
#HDMAD
A = np.loadtxt("Data/disease-microbe matrix.txt")  # 这是 HMDAD数据库 (39,292) 39是disease,292是 microbes
Sd_che = np.loadtxt("Data/dfs.txt") # (39,39) 疾病功能相似性矩阵
Sm_f = np.loadtxt("Data/mfs.txt")   #(292,292)  微生物功能相似性矩阵
Sd_dis = np.loadtxt("Data/d-dis.txt") #(39,39)
Sm_dis = np.loadtxt("Data/m-dis.txt") # (292,292)
known = np.loadtxt("Data/known1.txt")   # 450对 已知关系 # 已知关联索引（序号从1开始）
unknown = np.loadtxt("Data/unknown1.txt")  # 10938对未知关系  # 未知关联索引（序号从1开始）

# # # Disbiome
# # A = np.loadtxt("Data1/disease-microbe matrix.txt")  # 这是 HMDAD数据库 (39,292) 39是disease,292是 microbes
# # Sd_che = np.loadtxt("Data1/DFS.txt") # (39,39) 疾病功能相似性矩阵
# # Sm_f = np.loadtxt("Data1/MFS.txt")   #(292,292)  微生物功能相似性矩阵
# # Sd_dis = np.loadtxt("Data1/d-dis.txt") #(39,39)
# # Sm_dis = np.loadtxt("Data1/m-dis.txt") # (292,292)
# # known = np.loadtxt("Data1/known.txt")   # 450对 已知关系 # 已知关联索引（序号从1开始）
# # unknown = np.loadtxt("Data1/unknown.txt")  # 10938对未知关系  # 未知关联索引（序号从1开始）
#
#
# # MDAD
# # # 修改
# # A = np.loadtxt("MDAD/drug_microbe_matrix.txt")  #
# #
# # Sd_che = np.loadtxt("MDAD/drug_structure_sim.txt") #
# # Sm_f = np.loadtxt("MDAD/microbe_function_sim.txt")   #
# # # 放开
# # Sd_dis = np.loadtxt("MDAD/Sr_dis_matrix.txt")
# # Sm_dis = np.loadtxt("MDAD/Sm_dis_matrix.txt")
# # # 本来注释掉的
# # #Sd_dis = np.loadtxt("Data/d-dis.txt")
# # #Sm_dis = np.loadtxt("Data/m-dis.txt")
# # # Sd_dis = np.loadtxt("MDAD/Sr_dis_matrix.txt")
# # # Sm_dis = np.loadtxt("MDAD/Sm_dis_matrix.txt")
# # known = np.loadtxt("MDAD/known.txt")   #
# # unknown = np.loadtxt("MDAD/unknown.txt")  #
# #
#
#
TP,TN,FP,FN=0,0,0,0
scores=[]
tlabels=[]
#
# 5-fold cv
def kflod_5(num):
    k = []
    unk = []
    lk = len(known)  # 已知关联数 lk 450
    luk = len(unknown)  # 未知关联数 luk 10938
    for i in range(lk):
        k.append(i)      #k为已知关联
    for i in range(luk):
        unk.append(i)
    random.shuffle(k)  # 打乱顺序
    random.shuffle(unk)
    # for cv in range(1, 6):
    #     interaction = np.array(list(A))
    #     ratio = 3  # 负例和正例的比例，你可以自己试 1, 2, 3, 5 看效果
    #     if cv < 5:
    #         B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]  # 1/5的1的索引
    #         B2 = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]  # 1/5的0的索引
    #         for i in range(lk // 5):
    #             interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
    #     else:
    #         B1 = known[k[(cv - 1) * (lk // 5):lk], :]
    #         B2 = unknown[unk[(cv - 1) * (luk // 5):luk], :]
    #         for i in range(lk - (lk // 5)*4):
    #             interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
    ratio = 1 # <<< 你可以改这里

    for cv in range(1, 6):
        interaction = np.array(list(A))

        if cv < 5:
            B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]
            B2_raw = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]
        else:
            B1 = known[k[(cv - 1) * (lk // 5):lk], :]
            B2_raw = unknown[unk[(cv - 1) * (luk // 5):luk], :]

        # 下采样
        num_pos = B1.shape[0]
        num_neg_wanted = ratio * num_pos
        if B2_raw.shape[0] > num_neg_wanted:
            idx = np.random.choice(B2_raw.shape[0], num_neg_wanted, replace=False)
            B2 = B2_raw[idx]
        else:
            B2 = B2_raw

        # 后续不变
        for i in range(B1.shape[0]):
            interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0

        Sr_m_HIP = HIP_Calculate(interaction)  # 药物的HIP相似性 (39,39)
        Sm_r_HIP = HIP_Calculate(interaction.T)  # 微生物的HIP相似性 (292,292)
        Sm_r_GIP = GIP_Calculate(interaction)  # 微生物GIP相似性 (292,292)
        Sr_m_GIP = GIP_Calculate1(interaction)  # 药物GIP相似性 (39,39)
        # mfs1 = np.loadtxt("Data/mfs1.txt")
        # dfs = np.loadtxt("Data/dfs.txt")
        Sr_m = (Sr_m_HIP + Sr_m_GIP) / 2 # SD (39,39)
        Sm_r = (Sm_r_HIP + Sm_r_GIP) / 2   # SM (292,292)

        # #Learning node topology representations
        Net = Net_construct(Sr_m, Sm_r)    #HN  (39+292,39+292) =(331,331)
        # #HMDAD
        # np.savetxt("Data/net" +".txt", Net)
        # H = np.loadtxt("Data/net.txt")
        # dis
        # np.savetxt("Data1/net" +".txt", Net)
        # H = np.loadtxt("Data1/net.txt")
        # MDAD
        # np.savetxt("MDAD/net" +".txt", Net)
        # H = np.loadtxt("MDAD/net.txt")
        # 输入：A (39,292), Y (39,39), W (292,292)
        Y =Sr_m
        W =Sm_r
        H = construct_H(A, Y, W)
        Omega = get_observed_mask(A)
        E = nuclear_norm_minimization(H, Omega)
        A1= extract_predicted_A(E, A.shape)  # 得到预测矩阵 A_pred
        # A1 = np.power(A1, 0.3)  # γ校正
      #  np.savetxt("A1_50.txt", A1)
        #np.savetxt("A1_100.txt", A1)
      #  np.savetxt("A1_100.txt", A1)
        np.savetxt("A1.txt", A1)
    #    train33(Net,interaction,0)      #interaction (39,292)
        train33(Net, interaction, 0, sim_mat=Net)  # 第一个表示拓扑

        print("------------------------------------------")
        # topo_emb=topo_feature(Net)          #提取出的节点topology特征矩阵 （1373+173)*128
        # #Learning node attribute representations
        Sdd = RWR(Sr_m)       # SDMM    39，39
       # print("Sdd.shape",Sdd.shape)
        Smm = RWR(Sm_r)       # SMDD   292，292
       # print("Sdd.shape", Sdd.shape)
         #Hdmad
        np.savetxt("Data/Sdd.txt", Sdd)
        np.savetxt("Data/Smm.txt", Smm)
        #dis
        # np.savetxt("Data1/Sdd.txt", Sdd)
        # np.savetxt("Data1/Smm.txt", Smm)
        #MDAD
        # np.savetxt("MDAD/Sdd.txt", Sdd)
        # np.savetxt("MDAD/Smm.txt", Smm)
        A_r = np.hstack((np.hstack((np.hstack((interaction, Sd_che)), interaction)), Sdd))   #sdche 疾病功能相似性矩阵  (39,662)
        A_m = np.hstack((np.hstack((np.hstack((interaction.T, Sm_f)), interaction.T)), Smm))   # smf 微生物功能相似性矩阵 (292,662)

        # 修改打开，这个之前是关闭的

        A_r = np.hstack((A_r, Sd_dis))  #Sd_dis 是cos similarity
        A_m = np.hstack((A_m, Sm_dis))   #Sm_dis 是cos similarity
        # 为啥注释掉了 可能是效果不好了
        #train2(A_r, 0)
        # train2(A_m, 1)
        #interaction = (interaction + A1+A)/3
       # A2 = (A1+A)/2
        # construct Feature matrix for drug-microbe node pair
       # df, mf = Fmatrix(interaction)  # df:(39,1025) #mf(292,1025)
        # 修改
        df, mf = Fmatrix(interaction)
        min_max_scaler = preprocessing.MinMaxScaler()
        df = min_max_scaler.fit_transform(df)
        mf = min_max_scaler.fit_transform(mf)

        score = torch.sigmoid(torch.FloatTensor(np.dot(df, mf.T)))   # 这行代码的作用是计算两个矩阵的点积，并通过Sigmoid函数将结果映射到[0, 1]范围内
        # 转换为 DataFrame（默认保留行列结构）
        df = pd.DataFrame(score)

        # 保存为 Excel 文件（保持行列结构）
        df.to_excel('score.xlsx', index=False, header=False)  # 不加索引、不加列标题
        score = np.array(score)

        for i in range(len(B1)):
            index1 = int(B1[i, 0] - 1)
            index2 = int(B1[i, 1] - 1)
            scores.append(score[index1, index2])  # 计算的是测试集上的分数
            tlabels.append(A[index1, index2])
        for i in range(len(B2)):
            index1 = int(B2[i, 0] - 1)
            index2 = int(B2[i, 1] - 1)            # 计算的是测试集上的分数
            scores.append(score[index1, index2]) # 预测标签
            tlabels.append(A[index1, index2])   # 真实标签
        print("fold cv--{}".format(cv))
    fpr, tpr, threshold = roc_curve(tlabels, scores) # 计算ROC曲线
    num=str(num)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/fpr"+num+".txt",fpr)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/tpr"+num+".txt",tpr)
    np.savetxt("score GCAT=0.05SAE=0.01/score"+num+".txt",score)
    np.savetxt("tlable GCAT=0.05SAE=0.01/tlable"+num+".txt",tlabels)
    auc_val = auc(fpr,tpr)
    precision, recall, _ = precision_recall_curve(tlabels,scores)
    aupr_score = auc(recall, precision)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/precision" + num + ".txt",  precision)
    np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/recall" + num + ".txt", recall)
    # pred_labels = [1 if s >= 0.5 else 0 for s in scores]
    # print("Predicted label distribution:", sum(pred_labels), "/", len(pred_labels))
    #
    # acc = accuracy_score(tlabels, pred_labels )
    # mcc = matthews_corrcoef(tlabels, pred_labels )
    # f1 = f1_score(tlabels,pred_labels )
    print("auc:",auc_val)
    print("aupr:",aupr_score)
    # print("acc:",acc)
    # print("f1:",f1)
    # print("mcc:",mcc)
    return auc_val,aupr_score


auc_val=[]
aupr_val=[]
# acc_val=[]
# f1_score_val= []
# mcc_val=[]
for i in range(10):
    auc1,aupr=kflod_5(i)
    print("------------------------------")
    auc_val.append(auc1)
    aupr_val.append(aupr)
    # acc_val.append(acc)
    # f1_score_val.append(f1_score1)
    # mcc_val.append(mcc)
    np.savetxt("AUC--------CSAEtt/auc.txt",auc_val)
    np.savetxt("AUC--------CSAEtt/aupr.txt",aupr_val)
    # np.savetxt("AUC--------CSAEtt/acc.txt",acc_val)
    # np.savetxt("AUC--------CSAEtt/f1_score.txt",f1_score_val)
    # np.savetxt("AUC--------CSAEtt/mcc.txt", mcc_val)

print("auc_val:",sum(auc_val)/10)
print("aupr_val:",sum(aupr_val)/10)
# print("acc_val:",sum(acc_val)/10)
# print("f1_score_val:",sum(f1_score_val)/10)
# print("mcc_val:",sum(mcc_val)/10)
print("----------------------------------------------------------------------------------------------------------------")
print("训练完毕")

#####  2 折0.9746

#
# #
# # # 超参数分析
# import os
#
# # 设置保存根路径
# #BASE_SAVE_DIR = r"C:\Users\胡先俊\Desktop\论文\实验数据\超参数分析"
# # 设置保存根路径
# BASE_SAVE_DIR = r"C:\Users\胡先俊\Desktop\论文\实验数据\超参数分析\2"
# import numpy as np
# from CSEAE import *
#
# from Feature_matrix import *
# import random
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
# from sklearn import preprocessing
# from model1 import *
# # 原代码
# # ########## 已经能复现原来的性能效果了  0.97
# #HDMAD
# A = np.loadtxt("Data/disease-microbe matrix.txt")  # 这是 HMDAD数据库 (39,292) 39是disease,292是 microbes
# Sd_che = np.loadtxt("Data/dfs.txt") # (39,39) 疾病功能相似性矩阵
# Sm_f = np.loadtxt("Data/mfs.txt")   #(292,292)  微生物功能相似性矩阵
# Sd_dis = np.loadtxt("Data/d-dis.txt") #(39,39)
# Sm_dis = np.loadtxt("Data/m-dis.txt") # (292,292)
# known = np.loadtxt("Data/known1.txt")   # 450对 已知关系 # 已知关联索引（序号从1开始）
# unknown = np.loadtxt("Data/unknown1.txt")  # 10938对未知关系  # 未知关联索引（序号从1开始）
# #
# # # Disbiome
# # A = np.loadtxt("Data1/disease-microbe matrix.txt")  # 这是 HMDAD数据库 (39,292) 39是disease,292是 microbes
# # Sd_che = np.loadtxt("Data1/DFS.txt") # (39,39) 疾病功能相似性矩阵
# # Sm_f = np.loadtxt("Data1/MFS.txt")   #(292,292)  微生物功能相似性矩阵
# # Sd_dis = np.loadtxt("Data1/d-dis.txt") #(39,39)
# # Sm_dis = np.loadtxt("Data1/m-dis.txt") # (292,292)
# # known = np.loadtxt("Data1/known.txt")   # 450对 已知关系 # 已知关联索引（序号从1开始）
# # unknown = np.loadtxt("Data1/unknown.txt")  # 10938对未知关系  # 未知关联索引（序号从1开始）
# #
# #
# # #MDAD
# # # 修改
# # A = np.loadtxt("MDAD/drug_microbe_matrix.txt")  #
# #
# # Sd_che = np.loadtxt("MDAD/drug_structure_sim.txt") #
# # Sm_f = np.loadtxt("MDAD/microbe_function_sim.txt")   #
# # # 放开
# # Sd_dis = np.loadtxt("MDAD/Sr_dis_matrix.txt")
# # Sm_dis = np.loadtxt("MDAD/Sm_dis_matrix.txt")
# # # 本来注释掉的
# # #Sd_dis = np.loadtxt("Data/d-dis.txt")
# # #Sm_dis = np.loadtxt("Data/m-dis.txt")
# # # Sd_dis = np.loadtxt("MDAD/Sr_dis_matrix.txt")
# # # Sm_dis = np.loadtxt("MDAD/Sm_dis_matrix.txt")
# # known = np.loadtxt("MDAD/known.txt")   #
# # unknown = np.loadtxt("MDAD/unknown.txt")  #
# #
#
#
# TP,TN,FP,FN=0,0,0,0
# scores=[]
# tlabels=[]
#
#
# def kflod_5(num,alpha,beta):
#     k = []
#     unk = []
#     lk = len(known)  # 已知关联数 lk 450
#     luk = len(unknown)  # 未知关联数 luk 10938
#     for i in range(lk):
#         k.append(i)      #k为已知关联
#     for i in range(luk):
#         unk.append(i)
#     random.shuffle(k)  # 打乱顺序
#     random.shuffle(unk)
#     # for cv in range(1, 6):
#     #     interaction = np.array(list(A))
#     #     ratio = 3  # 负例和正例的比例，你可以自己试 1, 2, 3, 5 看效果
#     #     if cv < 5:
#     #         B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]  # 1/5的1的索引
#     #         B2 = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]  # 1/5的0的索引
#     #         for i in range(lk // 5):
#     #             interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
#     #     else:
#     #         B1 = known[k[(cv - 1) * (lk // 5):lk], :]
#     #         B2 = unknown[unk[(cv - 1) * (luk // 5):luk], :]
#     #         for i in range(lk - (lk // 5)*4):
#     #             interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
#     ratio = 1 # <<< 你可以改这里
#
#     for cv in range(1, 6):
#         interaction = np.array(list(A))
#
#         if cv < 5:
#             B1 = known[k[(cv - 1) * (lk // 5):(lk // 5) * cv], :]
#             B2_raw = unknown[unk[(cv - 1) * (luk // 5):(luk // 5) * cv], :]
#         else:
#             B1 = known[k[(cv - 1) * (lk // 5):lk], :]
#             B2_raw = unknown[unk[(cv - 1) * (luk // 5):luk], :]
#
#         # 下采样
#         num_pos = B1.shape[0]
#         num_neg_wanted = ratio * num_pos
#         if B2_raw.shape[0] > num_neg_wanted:
#             idx = np.random.choice(B2_raw.shape[0], num_neg_wanted, replace=False)
#             B2 = B2_raw[idx]
#         else:
#             B2 = B2_raw
#
#         # 后续不变
#         for i in range(B1.shape[0]):
#             interaction[int(B1[i, 0]) - 1, int(B1[i, 1]) - 1] = 0
#
#         Sr_m_HIP = HIP_Calculate(interaction)  # 药物的HIP相似性 (39,39)
#         Sm_r_HIP = HIP_Calculate(interaction.T)  # 微生物的HIP相似性 (292,292)
#         Sm_r_GIP = GIP_Calculate(interaction)  # 微生物GIP相似性 (292,292)
#         Sr_m_GIP = GIP_Calculate1(interaction)  # 药物GIP相似性 (39,39)
#         # mfs1 = np.loadtxt("Data/mfs1.txt")
#         # dfs = np.loadtxt("Data/dfs.txt")
#         Sr_m = (Sr_m_HIP + Sr_m_GIP) / 2 # SD (39,39)
#         Sm_r = (Sm_r_HIP + Sm_r_GIP) / 2   # SM (292,292)
#
#         # #Learning node topology representations
#         Net = Net_construct(Sr_m, Sm_r)    #HN  (39+292,39+292) =(331,331)
#         # #HMDAD
#         # np.savetxt("Data/net" +".txt", Net)
#         # H = np.loadtxt("Data/net.txt")
#         # dis
#         # np.savetxt("Data1/net" +".txt", Net)
#         # H = np.loadtxt("Data1/net.txt")
#         # MDAD
#         # np.savetxt("MDAD/net" +".txt", Net)
#         # H = np.loadtxt("MDAD/net.txt")
#         # 输入：A (39,292), Y (39,39), W (292,292)
#         Y =Sr_m
#         W =Sm_r
#         H = construct_H(A, Y, W)
#         Omega = get_observed_mask(A)
#      #   E = nuclear_norm_minimization(H, Omega,omega,gamma)
#         E = nuclear_norm_minimization(H, Omega)
#         A1= extract_predicted_A(E, A.shape)  # 得到预测矩阵 A_pred (39,292)
#         np.savetxt("A1.txt", A1)
#     #    train33(Net,interaction,0)      #interaction (39,292)
#         train33(Net, interaction, 0, sim_mat=Net,alpha= alpha, beta= beta)  # 第一个表示拓扑
#
#         print("------------------------------------------")
#         # topo_emb=topo_feature(Net)          #提取出的节点topology特征矩阵 （1373+173)*128
#         # #Learning node attribute representations
#         Sdd = RWR(Sr_m)       # SDMM    39，39
#        # print("Sdd.shape",Sdd.shape)
#         Smm = RWR(Sm_r)       # SMDD   292，292
#        # print("Sdd.shape", Sdd.shape)
#          #Hdmad
#         np.savetxt("Data/Sdd.txt", Sdd)
#         np.savetxt("Data/Smm.txt", Smm)
#         #dis
#         # np.savetxt("Data1/Sdd.txt", Sdd)
#         # np.savetxt("Data1/Smm.txt", Smm)
#         #MDAD
#         # np.savetxt("MDAD/Sdd.txt", Sdd)
#         # np.savetxt("MDAD/Smm.txt", Smm)
#         A_r = np.hstack((np.hstack((np.hstack((interaction, Sd_che)), interaction)), Sdd))   #sdche 疾病功能相似性矩阵  (39,662)
#         A_m = np.hstack((np.hstack((np.hstack((interaction.T, Sm_f)), interaction.T)), Smm))   # smf 微生物功能相似性矩阵 (292,662)
#
#         # 修改打开，这个之前是关闭的
#
#         A_r = np.hstack((A_r, Sd_dis))  #Sd_dis 是cos similarity
#         A_m = np.hstack((A_m, Sm_dis))   #Sm_dis 是cos similarity
#         # 为啥注释掉了 可能是效果不好了
#         train2(A_r, 0)
#         train2(A_m, 1)
#         #interaction = (interaction + A1+A)/3
#        # A2 = (A1+A)/2
#         # construct Feature matrix for drug-microbe node pair
#        # df, mf = Fmatrix(interaction)  # df:(39,1025) #mf(292,1025)
#         # 修改
#         df, mf = Fmatrix(A)
#         min_max_scaler = preprocessing.MinMaxScaler()
#         df = min_max_scaler.fit_transform(df)
#         mf = min_max_scaler.fit_transform(mf)
#         score = torch.sigmoid(torch.FloatTensor(np.dot(df, mf.T)))   # 这行代码的作用是计算两个矩阵的点积，并通过Sigmoid函数将结果映射到[0, 1]范围内
#         score = np.array(score)
#         for i in range(len(B1)):
#             index1 = int(B1[i, 0] - 1)
#             index2 = int(B1[i, 1] - 1)
#             scores.append(score[index1, index2])
#             tlabels.append(A[index1, index2])
#         for i in range(len(B2)):
#             index1 = int(B2[i, 0] - 1)
#             index2 = int(B2[i, 1] - 1)
#             scores.append(score[index1, index2]) # 预测标签
#             tlabels.append(A[index1, index2])   # 真实标签
#         print("fold cv--{}".format(cv))
#     fpr, tpr, threshold = roc_curve(tlabels, scores) # 计算ROC曲线
#     num=str(num)
#     save_dir = os.path.join(BASE_SAVE_DIR, f"alpha={alpha}_beta={beta}")
#     os.makedirs(save_dir, exist_ok=True)
#
#     np.savetxt(os.path.join(save_dir, f"fpr{num}.txt"), fpr)
#     np.savetxt(os.path.join(save_dir, f"tpr{num}.txt"), tpr)
#     np.savetxt(os.path.join(save_dir, f"score{num}.txt"), score)
#     np.savetxt(os.path.join(save_dir, f"label{num}.txt"), tlabels)
#
#
#     # np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/fpr"+num+".txt",fpr)
#     # np.savetxt("fpr_tpr GCAT=0.05SAE=0.01/tpr"+num+".txt",tpr)
#     # np.savetxt("score GCAT=0.05SAE=0.01/score"+num+".txt",score)
#     # np.savetxt("tlable GCAT=0.05SAE=0.01/tlable"+num+".txt",tlabels)
#     auc_val = auc(fpr,tpr)
#     precision, recall, _ = precision_recall_curve(tlabels,scores)
#     aupr_score = auc(recall, precision)
#     # np.savetxt("HMDAD_roc_pr/precision" + num + ".txt",  precision)
#     # np.savetxt("HMDAD_roc_pr/recall" + num + ".txt", recall)
#     np.savetxt(os.path.join(save_dir, f"precision{num}.txt"), precision)
#     np.savetxt(os.path.join(save_dir, f"recall{num}.txt"), recall)
#     print("auc:",auc_val)
#     print("aupr:",aupr_score)
#
#     return auc_val,aupr_score
#
# #
# # auc_val=[]
# # aupr_val=[]
# #
# # for i in range(10):
# #     auc1,aupr=kflod_5(i)
# #     print("------------------------------")
# #     auc_val.append(auc1)
# #     aupr_val.append(aupr)
# #
# #     np.savetxt("AUC--------CSAEtt/auc.txt",auc_val)
# #     np.savetxt("AUC--------CSAEtt/aupr.txt",aupr_val)
# #
# #
# # print("auc_val:",sum(auc_val)/10)
# # print("aupr_val:",sum(aupr_val)/10)
# #
# # print("----------------------------------------------------------------------------------------------------------------")
# # print("训练完毕")
# #
# # from itertools import product
# #
# # omega_list = [1, 10, 100,1000]
# # gamma_list = [1, 10, 100,1000]
# #
# # for omega, gamma in product(omega_list, gamma_list):
# #     auc_val = []
# #     aupr_val = []
# #     for i in range(10):
# #         auc1, aupr = kflod_5(i, omega=omega, gamma=gamma)
# #         auc_val.append(auc1)
# #         aupr_val.append(aupr)
# #     # 这个是第一组那个NNM的参数
# #     summary_dir = os.path.join(BASE_SAVE_DIR, f"omega={omega}_gamma={gamma}")
# #     os.makedirs(summary_dir, exist_ok=True)
# #     np.savetxt(os.path.join(summary_dir, "auc_summary.txt"), auc_val)
# #     np.savetxt(os.path.join(summary_dir, "aupr_summary.txt"), aupr_val)
# #     print(f"[omega={omega}, gamma={gamma}] AUC={np.mean(auc_val):.4f}, AUPR={np.mean(aupr_val):.4f}")
# #
# #
# # from itertools import product
# #
# # omega_list = [1, 10, 100,1000]
# # gamma_list = [1, 10, 100,1000]
# #
# # for omega, gamma in product(omega_list, gamma_list):
# #     auc_val = []
# #     aupr_val = []
# #     for i in range(10):
# #         auc1, aupr = kflod_5(i, omega=omega, gamma=gamma)
# #         auc_val.append(auc1)
# #         aupr_val.append(aupr)
# #     # 这个是第一组那个NNM的参数
# #     summary_dir = os.path.join(BASE_SAVE_DIR, f"omega={omega}_gamma={gamma}")
# #     os.makedirs(summary_dir, exist_ok=True)
# #     np.savetxt(os.path.join(summary_dir, "auc_summary.txt"), auc_val)
# #     np.savetxt(os.path.join(summary_dir, "aupr_summary.txt"), aupr_val)
# #     print(f"[omega={omega}, gamma={gamma}] AUC={np.mean(auc_val):.4f}, AUPR={np.mean(aupr_val):.4f}")
# #
#
#
# from itertools import product
#
# omega_list = [0.01,1, 10,50,100]
# gamma_list = [0.01,1, 10,50,100]
#
# for alpha, beta in product(omega_list, gamma_list):
#     auc_val = []
#     aupr_val = []
#     for i in range(10):
#         auc1, aupr = kflod_5(i, alpha=alpha, beta=beta)
#         auc_val.append(auc1)
#         aupr_val.append(aupr)
#     # 这个是第一组那个NNM的参数
#     summary_dir = os.path.join(BASE_SAVE_DIR, f"alpha={alpha}_beta={beta}")
#     os.makedirs(summary_dir, exist_ok=True)
#     np.savetxt(os.path.join(summary_dir, "auc_summary.txt"), auc_val)
#     np.savetxt(os.path.join(summary_dir, "aupr_summary.txt"), aupr_val)
#     print(f"[omega={alpha}, gamma={beta}] AUC={np.mean(auc_val):.4f}, AUPR={np.mean(aupr_val):.4f}")
#
#
