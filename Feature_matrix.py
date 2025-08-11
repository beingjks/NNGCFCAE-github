import numpy as np
import pandas as pd


# A2 = np.loadtxt("Data/disease-microbe matrix.txt")
#HMDAD
def Fmatrix(M):
    A1 = np.loadtxt("A1.txt")
    A = M
    #A1 =A
    A1= (A1+A)/2


    """
   # Z = np.loadtxt("Data/topo embedding1.txt")  # 节点拓扑表示矩阵
   #  Z_d = Z[0:39, :]  # 药物节点拓扑表示矩阵
   #  Z_m = Z[39:331, :]  # 微生物节点拓扑表示矩阵
   #  为啥注释掉了 这和文章里说的不一样了。

    """



    Z = np.loadtxt("Data/topo embedding1.txt")  # 节点拓扑表示矩阵
    Z_d = Z[0:39, :]  # 药物节点拓扑表示矩阵
    Z_m = Z[39:331, :]  # 微生物节点拓扑表示矩阵
    A_dd = np.loadtxt("Data/attr embedding_d.txt")  # 药物节点属性表示矩阵
    A_mm = np.loadtxt("Data/attr embedding_m.txt")  # 微生物节点属性表示矩阵
    Sd_che = np.loadtxt("Data/dfs.txt")  # S_r^Che drug structure similarity
    Sm_fun = np.loadtxt("Data/mfs.txt")  # S_m^f microbe function similarity
    Sd_dis = np.loadtxt("Data/d-dis.txt")
    Sm_dis = np.loadtxt("Data/m-dis.txt")
    Sdd = np.loadtxt("Data/Sdd.txt")
    Smm = np.loadtxt("Data/Smm.txt")
    # # disbiome
    # Z = np.loadtxt("Data1/topo embedding1.txt")  # 节点拓扑表示矩阵
    # Z_d = Z[0:218, :]  # 药物节点拓扑表示矩阵
    # Z_m = Z[218:1270, :]  # 微生物节点拓扑表示矩阵
    # A_dd = np.loadtxt("Data1/attr embedding_d.txt")  # 药物节点属性表示矩阵
    # A_mm = np.loadtxt("Data1/attr embedding_m.txt")  # 微生物节点属性表示矩阵
    # Sd_che = np.loadtxt("Data1/dfs.txt")  # S_r^Che drug structure similarity
    # Sm_fun = np.loadtxt("Data1/mfs.txt")  # S_m^f microbe function similarity
    # Sd_dis = np.loadtxt("Data1/d-dis.txt")
    # Sm_dis = np.loadtxt("Data1/m-dis.txt")
    # Sdd = np.loadtxt("Data1/Sdd.txt")
    # Smm = np.loadtxt("Data1/Smm.txt")

    # print("Sdd.shape:", Sdd.shape)
    # print("A.shape:", A.shape)
    Disease_feature = np.hstack((Z_d, np.hstack((A_dd,np.hstack((Sd_che, A))))))
 #   Disease_feature = np.hstack((Z_d, np.hstack((A_dd,np.hstack((Sd_che, A))))))
   # print("Disease_feature.shape:", Disease_feature.shape)
  # Disease_feature =  np.hstack((A_dd,np.hstack((Sd_che, A))))    # (AD,SDFS,A,SDCOS,A,SDMM,A)  # 我的修改
   # Disease_feature =  np.hstack((Z_d, np.hstack((Sd_che, A))))
    Disease_feature = np.hstack((np.hstack((Disease_feature, Sd_dis)), A))
    Disease_feature=np.hstack((np.hstack((Disease_feature,Sdd)),A1))
  #  Disease_feature = np.hstack((np.hstack((Disease_feature, A1)),A[:,0:39]))

    Microbe_feature = np.hstack((Z_m, np.hstack((A_mm,np.hstack((A.T, Sm_fun))))))    # 我的修改
   # Microbe_feature = np.hstack((Z_m, np.hstack((A_mm,np.hstack((A.T, Sm_fun))))))
    # Microbe_feature =  np.hstack((Z_m,np.hstack((A.T, Sm_fun))))
    Microbe_feature = np.hstack((np.hstack((Microbe_feature,A.T)),Sm_dis))
    Microbe_feature=np.hstack((np.hstack((Microbe_feature,A1.T)),Smm))
  #  Microbe_feature = np.hstack((np.hstack((Microbe_feature, A.T[:,0:292])),A1.T))
    return  Disease_feature,Microbe_feature

#
# #disbiome
# def Fmatrix(M):
#     A1 = np.loadtxt("A1.txt")
#     A = M
#     A1 = (0.2*A1 + 0.8*A)
#
#     """
#    # Z = np.loadtxt("Data/topo embedding1.txt")  # 节点拓扑表示矩阵
#    #  Z_d = Z[0:39, :]  # 药物节点拓扑表示矩阵
#    #  Z_m = Z[39:331, :]  # 微生物节点拓扑表示矩阵
#    #  为啥注释掉了 这和文章里说的不一样了。
#
#     """
#
#
#
#     # disbiome
#     Z = np.loadtxt("Data1/topo embedding1.txt")  # 节点拓扑表示矩阵
#     Z_d = Z[0:218, :]  # 药物节点拓扑表示矩阵
#     Z_m = Z[218:1270, :]  # 微生物节点拓扑表示矩阵
#     A_dd = np.loadtxt("Data1/attr embedding_d.txt")  # 药物节点属性表示矩阵
#     A_mm = np.loadtxt("Data1/attr embedding_m.txt")  # 微生物节点属性表示矩阵
#     Sd_che = np.loadtxt("Data1/dfs.txt")  # S_r^Che drug structure similarity
#     Sm_fun = np.loadtxt("Data1/mfs.txt")  # S_m^f microbe function similarity
#     Sd_dis = np.loadtxt("Data1/d-dis.txt")
#     Sm_dis = np.loadtxt("Data1/m-dis.txt")
#     Sdd = np.loadtxt("Data1/Sdd.txt")
#     Smm = np.loadtxt("Data1/Smm.txt")
#
#     # print("Sdd.shape:", Sdd.shape)
#     # print("A.shape:", A.shape)
#     #Disease_feature = np.hstack((Z_d, np.hstack((A_dd,np.hstack((Sd_che, A))))))
#     Disease_feature = np.hstack((Z_d, np.hstack((A_dd,np.hstack((Sd_che, A))))))
#    # print("Disease_feature.shape:", Disease_feature.shape)
#   #  Disease_feature =  np.hstack((A_dd,np.hstack((Sd_che, A))))    # (AD,SDFS,A,SDCOS,A,SDMM,A)  # 我的修改
#     Disease_feature = np.hstack((np.hstack((Disease_feature, Sd_dis)), A))
#     Disease_feature=np.hstack((np.hstack((Disease_feature,Sdd)),A1))
#   #  Disease_feature = np.hstack((np.hstack((Disease_feature, A1)),A[:,0:39]))
#
#    # Microbe_feature = np.hstack((Z_m, np.hstack((A_mm,np.hstack((A.T, Sm_fun))))))    # 我的修改
#     Microbe_feature = np.hstack((Z_m, np.hstack((A_mm,np.hstack((A.T, Sm_fun))))))
#    # Microbe_feature =  np.hstack((A_mm,np.hstack((A.T, Sm_fun))))
#     Microbe_feature = np.hstack((np.hstack((Microbe_feature,A.T)),Sm_dis))
#     Microbe_feature=np.hstack((np.hstack((Microbe_feature,A1.T)),Smm))
#   #  Microbe_feature = np.hstack((np.hstack((Microbe_feature, A.T[:,0:292])),A1.T))
#     return  Disease_feature,Microbe_feature


#MDAD

# def Fmatrix(M):
#     A=M
#     A1 = np.loadtxt("A1.txt")
#
#     # Z = np.loadtxt("Data/topo embedding1.txt")  # 节点拓扑表示矩阵
#    #  Z_d = Z[0:39, :]  # 药物节点拓扑表示矩阵
#    #  Z_m = Z[39:331, :]  # 微生物节点拓扑表示矩阵
#    #  为啥注释掉了 这和文章里说的不一样了。
#
#
#     Z = np.loadtxt("MDAD/topo embedding1.txt")  # 节点拓扑表示矩阵
#     Z_d = Z[0:1373, :]  # 药物节点拓扑表示矩阵
#     Z_m = Z[1373:1546, :]  # 微生物节点拓扑表示矩阵
#     A_dd = np.loadtxt("MDAD/attr embedding_d.txt")  # 药物节点属性表示矩阵
#     A_mm = np.loadtxt("MDAD/attr embedding_m.txt")  # 微生物节点属性表示矩阵
#     Sd_che = np.loadtxt("MDAD/drug_structure_sim.txt")  # S_r^Che drug structure similarity
#     Sm_fun = np.loadtxt("MDAD/microbe_function_sim.txt")  # S_m^f microbe function similarity
#     Sd_dis=np.loadtxt("MDAD/Sr_dis_matrix.txt")
#     Sm_dis=np.loadtxt("MDAD/Sm_dis_matrix.txt")
#
#     Sdd=np.loadtxt("MDAD/Srr.txt")
#     Smm=np.loadtxt("MDAD/Smm.txt")
#     #Disease_feature = np.hstack((Z_d, np.hstack((A_dd,np.hstack((Sd_che, A))))))
#     Disease_feature = np.hstack((Z_d, np.hstack((A_dd,np.hstack((Sd_che, A))))))
#    # Disease_feature =  np.hstack((A_dd,np.hstack((Sd_che, A))))    # (AD,SDFS,A,SDCOS,A,SDMM,A)  # 我的修改
#     Disease_feature = np.hstack((np.hstack((Disease_feature, Sd_dis)), A))
#     Disease_feature=np.hstack((np.hstack((Disease_feature,Sdd)),A1))
#    # Microbe_feature = np.hstack((Z_m, np.hstack((A_mm,np.hstack((A.T, Sm_fun))))))    # 我的修改
#     Microbe_feature = np.hstack((Z_m, np.hstack((A_mm,np.hstack((A.T, Sm_fun))))))
#
#    # Microbe_feature =  np.hstack((A_mm,np.hstack((A.T, Sm_fun))))
#     Microbe_feature = np.hstack((np.hstack((Microbe_feature,A.T)),Sm_dis))
#     Microbe_feature=np.hstack((np.hstack((Microbe_feature,A1.T)),Smm))
#
#     return  Disease_feature,Microbe_feature


