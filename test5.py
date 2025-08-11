# import matplotlib.pyplot as plt
# from sklearn.metrics import auc
# import numpy as np
#
# # 假设以下为模型的 recall 和 precision 值（需要你用真实数据替换）
# recall_gsamda = np.loadtxt("HMDAD_roc_pr/GSAMDArecall.txt")
# precision_gsamda = np.loadtxt("HMDAD_roc_pr/GSAMDAprecision.txt")
#
# recall_mdasae = np.loadtxt("HMDAD_roc_pr/MDASAEre.txt")
# precision_mdasae = np.loadtxt("HMDAD_roc_pr/MDASAEpr.txt")
#
#
# recall_lagcn = np.loadtxt("HMDAD_roc_pr/LAGCNx_PR00.txt")
# precision_lagcn = np.loadtxt("HMDAD_roc_pr/LAGCNy_PR00.txt")
#
# recall_gvnnvae = np.loadtxt("HMDAD_roc_pr/MDAD_GVNNSAE_recall.txt")
# precision_gvnnvae = np.loadtxt("HMDAD_roc_pr/MDAD_GVNNSAE_precision.txt")
#
# recall_gcnmda = np.loadtxt("HMDAD_roc_pr/GCNMDArecall.txt")
# precision_gcnmda = np.loadtxt("HMDAD_roc_pr/GCNMDAprecision.txt")
#
# recall_nngcfcae = np.loadtxt("HMDAD_roc_pr/recall0_best2.txt")
# precision_nngcfcae = np.loadtxt("HMDAD_roc_pr/precision0_best2.txt")
# # 计算 AUPR
# aupr_nngcfcae = auc(recall_nngcfcae, precision_nngcfcae )
# aupr_mdasae = auc(recall_mdasae, precision_mdasae)
# aupr_gsamda = auc(recall_gsamda, precision_gsamda)
# aupr_lagcn= auc(recall_lagcn, precision_lagcn)
# aupr_gvnnvae= auc(recall_gvnnvae,precision_gvnnvae)
# aupr_gcnmda = auc(recall_gcnmda, precision_gcnmda)
#
# # 画图
# plt.figure(figsize=(10, 6))
#
# plt.plot(recall_mdasae, precision_mdasae, label=f'MDASAE (AUPR={aupr_mdasae:.4f})', color='darkorange')
# plt.plot(recall_gsamda, precision_gsamda, label=f'GSAMDA (AUPR={aupr_gsamda:.4f})', color='green')
# plt.plot(recall_gvnnvae , precision_gvnnvae , label=f'GVNNVAE (AUPR={aupr_gvnnvae:.4f})', color='mediumpurple')
# plt.plot(recall_lagcn, precision_lagcn,label=f'LAGCN (AUPR={aupr_lagcn:.4f})', color='brown')
# plt.plot(recall_gcnmda, precision_gcnmda, label=f'GCNMDA (AUPR={aupr_gcnmda:.4f})', color='crimson')
# plt.plot(recall_nngcfcae, precision_nngcfcae , label=f'NNGCFCAE(AUPR={aupr_nngcfcae:.4f})', color='cornflowerblue')
# plt.title("(b)PR Curves of HMDAD")
# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.legend(loc='lower left')
# plt.grid(True)
# plt.show()

#

import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np

# 假设以下为模型的 recall 和 precision 值（需要你用真实数据替换）
recall_gsamda = np.loadtxt("Dis_roc_pr/recallGSAMDA.txt")
precision_gsamda = np.loadtxt("Dis_roc_pr/precisionGSAMDA.txt")

recall_mdasae = np.loadtxt("Dis_roc_pr/MDASAEre.txt")
precision_mdasae = np.loadtxt("Dis_roc_pr/MDASAEpr.txt")


recall_lagcn = np.loadtxt("Dis_roc_pr/LAGCNx_PR.txt")
precision_lagcn = np.loadtxt("Dis_roc_pr/LAGCNy_PR.txt")

recall_gvnnvae = np.loadtxt("Dis_roc_pr/AB_GVNNVAE_recall0.txt")
precision_gvnnvae = np.loadtxt("Dis_roc_pr/AB_GVNNVAE_precision0.txt")

recall_gcnmda = np.loadtxt("Dis_roc_pr/GCNMDArecall.txt")
precision_gcnmda = np.loadtxt("Dis_roc_pr/GCNMDAprecision.txt")
#
# recall_nngcfcae = np.loadtxt("Dis_roc_pr/recall9.txt")
# precision_nngcfcae = np.loadtxt("Dis_roc_pr/precision9.txt")
recall_nngcfcae = np.loadtxt("Dis_roc_pr/recall0_best.txt")
precision_nngcfcae = np.loadtxt("Dis_roc_pr/precision0_best.txt")
# 计算 AUPR
aupr_nngcfcae = auc(recall_nngcfcae, precision_nngcfcae )
aupr_mdasae = auc(recall_mdasae, precision_mdasae)
aupr_gsamda = auc(recall_gsamda, precision_gsamda)
aupr_lagcn= auc(recall_lagcn, precision_lagcn)
aupr_gvnnvae= auc(recall_gvnnvae,precision_gvnnvae)
aupr_gcnmda = auc(recall_gcnmda, precision_gcnmda)

# 画图
plt.figure(figsize=(10, 6))

plt.plot(recall_mdasae, precision_mdasae, label=f'MDASAE (AUPR={aupr_mdasae:.4f})', color='darkorange')
plt.plot(recall_gsamda, precision_gsamda, label=f'GSAMDA (AUPR={aupr_gsamda:.4f})', color='green')
plt.plot(recall_gvnnvae , precision_gvnnvae , label=f'GVNNVAE (AUPR={aupr_gvnnvae:.4f})', color='mediumpurple')
plt.plot(recall_lagcn, precision_lagcn,label=f'LAGCN (AUPR={aupr_lagcn:.4f})', color='brown')
plt.plot(recall_gcnmda, precision_gcnmda, label=f'GCNMDA (AUPR={aupr_gcnmda:.4f})', color='crimson')
plt.plot(recall_nngcfcae, precision_nngcfcae , label=f'NNGCFCAE(AUPR={aupr_nngcfcae:.4f})', color='cornflowerblue')
plt.title("(b)PR Curves of Disbiome")
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
