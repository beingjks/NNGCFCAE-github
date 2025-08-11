# # ROC 曲线
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import auc
#
# # 假设这些是假定的 FPR 和 TPR 值（需要你用真实数据替换）
# fpr_gsamda = np.loadtxt("HMDAD_roc_pr/GSAMDAfpr.txt")
# tpr_gsamda = np.loadtxt("HMDAD_roc_pr/GSAMDAtpr.txt")
#
# fpr_gvnnvae = np.loadtxt("HMDAD_roc_pr/GVNNSAE_fpr6.txt")
# tpr_gvnnvae =  np.loadtxt("HMDAD_roc_pr/GVNNSAE_tpr6.txt")
#
# fpr_gcnmda = np.loadtxt("HMDAD_roc_pr/GCNMDAfpr.txt")
# tpr_gcnmda = np.loadtxt("HMDAD_roc_pr/GCNMDAtpr.txt")
#
# fpr_mdasae = np.loadtxt("HMDAD_roc_pr/MDASAEfpr.txt")
# tpr_mdasae =np.loadtxt("HMDAD_roc_pr/MDASAEtpr.txt")
#
# fpr_lagcn = np.loadtxt("HMDAD_roc_pr/LAGCNx_ROC.txt")
# tpr_lagcn = np.loadtxt("HMDAD_roc_pr/LAGCNy_ROC0.txt")
#
# fpr_nngcfcae = np.loadtxt("HMDAD_roc_pr/fpr0_best2.txt")
# tpr_nngcfcae = np.loadtxt("HMDAD_roc_pr/tpr0_best2.txt")
#
# # 计算AUC
# auc_gsamda = auc(fpr_gsamda, tpr_gsamda)
# auc_gvnnvae = auc(fpr_gvnnvae, tpr_gvnnvae)
# auc_gcnmda = auc(fpr_gcnmda, tpr_gcnmda)
# auc_mdasae = auc(fpr_mdasae, tpr_mdasae)
# auc_lagcn = auc(fpr_lagcn, tpr_lagcn)
# auc_nngcfcae = auc(fpr_nngcfcae, tpr_nngcfcae)
#
# # 绘制曲线
# plt.figure(figsize=(10, 6))
# plt.plot(fpr_gsamda, tpr_gsamda, color='blue', label=f'GSAMDA (AUC = {auc_gsamda:.4f})')
# plt.plot(fpr_gvnnvae, tpr_gvnnvae, color='orange', label=f'GVNNSAE (AUC = {auc_gvnnvae:.4f})')
# plt.plot(fpr_gcnmda, tpr_gcnmda, color='green', label=f'GCNMDA (AUC = {auc_gcnmda:.4f})')
# plt.plot(fpr_mdasae, tpr_mdasae, color='black', label=f'MDASAE (AUC = {auc_mdasae:.4f})')
# plt.plot(fpr_lagcn, tpr_lagcn, color='purple', label=f'LAGCN (AUC = {auc_lagcn:.4f})')
# plt.plot(fpr_nngcfcae,tpr_nngcfcae, color='red', label=f'NNGCFCAE (AUC = {auc_nngcfcae:.4f})')
#
# # 图形设置
# plt.title("ROC curves of HMDAD")
# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.show()
#
#
#
#
#
# # ROC 曲线
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import auc
#
# # 假设这些是假定的 FPR 和 TPR 值（需要你用真实数据替换）
# fpr_gsamda = np.loadtxt("Dis_roc_pr/GSAMDAfpr.txt")
# tpr_gsamda = np.loadtxt("Dis_roc_pr/GSAMDAtpr.txt")
#
# fpr_gvnnvae = np.loadtxt("Dis_roc_pr/NTSHMDAfpr.txt")
# tpr_gvnnvae =  np.loadtxt("Dis_roc_pr/NTSHMDAtpr.txt")
#
# fpr_gcnmda = np.loadtxt("Dis_roc_pr/GCNMDAfpr.txt")
# tpr_gcnmda = np.loadtxt("Dis_roc_pr/GCNMDAtpr.txt")
#
# fpr_mdasae = np.loadtxt("Dis_roc_pr/MDASAEfpr.txt")
# tpr_mdasae =np.loadtxt("Dis_roc_pr/MDASAEtpr.txt")
#
# fpr_lagcn = np.loadtxt("Dis_roc_pr/LAGCNx_ROC.txt")
# tpr_lagcn = np.loadtxt("Dis_roc_pr/LAGCNy_ROC.txt")
#
# # fpr_nngcfcae = np.loadtxt("Dis_roc_pr/fpr0_best2.txt")
# # tpr_nngcfcae = np.loadtxt("Dis_roc_pr/tpr0_best2.txt")
# fpr_nngcfcae = np.loadtxt("Dis_roc_pr/fpr0_best.txt")
# tpr_nngcfcae = np.loadtxt("Dis_roc_pr/tpr0_best.txt")
# # 计算AUC
# auc_gsamda = auc(fpr_gsamda, tpr_gsamda)
# auc_gvnnvae = auc(fpr_gvnnvae, tpr_gvnnvae)
# auc_gcnmda = auc(fpr_gcnmda, tpr_gcnmda)
# auc_mdasae = auc(fpr_mdasae, tpr_mdasae)
# auc_lagcn = auc(fpr_lagcn, tpr_lagcn)
# auc_nngcfcae = auc(fpr_nngcfcae, tpr_nngcfcae)
#
# # 绘制曲线
# plt.figure(figsize=(10, 6))
# plt.plot(fpr_gsamda, tpr_gsamda, color='blue', label=f'GSAMDA (AUC = {auc_gsamda:.4f})')
# plt.plot(fpr_gvnnvae, tpr_gvnnvae, color='orange', label=f'GVNNSAE (AUC = {auc_gvnnvae:.4f})')
# plt.plot(fpr_gcnmda, tpr_gcnmda, color='green', label=f'GCNMDA (AUC = {auc_gcnmda:.4f})')
# plt.plot(fpr_mdasae, tpr_mdasae, color='black', label=f'MDASAE (AUC = {auc_mdasae:.4f})')
# plt.plot(fpr_lagcn, tpr_lagcn, color='purple', label=f'LAGCN (AUC = {auc_lagcn:.4f})')
# plt.plot(fpr_nngcfcae,tpr_nngcfcae, color='red', label=f'NNGCFCAE (AUC = {auc_nngcfcae:.4f})')
#
# # 图形设置
# plt.title("ROC curves of Disbiome")
# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.show()
#
#
#
#


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# ————— 加载并计算 HMDAD 的 ROC/AUC —————
fpr_h = {
    'GSAMDA': np.loadtxt("HMDAD_roc_pr/GSAMDAfpr.txt"),
    'GVNNSAE': np.loadtxt("HMDAD_roc_pr/GVNNSAE_fpr6.txt"),
    'GCNMDA': np.loadtxt("HMDAD_roc_pr/GCNMDAfpr.txt"),
    'MDASAE': np.loadtxt("HMDAD_roc_pr/MDASAEfpr.txt"),
    'LAGCN': np.loadtxt("HMDAD_roc_pr/LAGCNx_ROC.txt"),
    'NNGCFCAE': np.loadtxt("HMDAD_roc_pr/fpr0_best2.txt")
}
tpr_h = {
    'GSAMDA': np.loadtxt("HMDAD_roc_pr/GSAMDAtpr.txt"),
    'GVNNSAE': np.loadtxt("HMDAD_roc_pr/GVNNSAE_tpr6.txt"),
    'GCNMDA': np.loadtxt("HMDAD_roc_pr/GCNMDAtpr.txt"),
    'MDASAE': np.loadtxt("HMDAD_roc_pr/MDASAEtpr.txt"),
    'LAGCN': np.loadtxt("HMDAD_roc_pr/LAGCNy_ROC0.txt"),
    'NNGCFCAE': np.loadtxt("HMDAD_roc_pr/tpr0_best2.txt")
}
auc_h = {name: auc(fpr_h[name], tpr_h[name]) for name in fpr_h}

# ————— 加载并计算 Disbiome 的 ROC/AUC —————
fpr_d = {
    'GSAMDA': np.loadtxt("Dis_roc_pr/GSAMDAfpr.txt"),
    'NTSHMDA': np.loadtxt("Dis_roc_pr/NTSHMDAfpr.txt"),
    'GCNMDA': np.loadtxt("Dis_roc_pr/GCNMDAfpr.txt"),
    'MDASAE': np.loadtxt("Dis_roc_pr/MDASAEfpr.txt"),
    'LAGCN': np.loadtxt("Dis_roc_pr/LAGCNx_ROC.txt"),
    'NNGCFCAE': np.loadtxt("Dis_roc_pr/fpr0_best.txt")
}
tpr_d = {
    'GSAMDA': np.loadtxt("Dis_roc_pr/GSAMDAtpr.txt"),
    'NTSHMDA': np.loadtxt("Dis_roc_pr/NTSHMDAtpr.txt"),
    'GCNMDA': np.loadtxt("Dis_roc_pr/GCNMDAtpr.txt"),
    'MDASAE': np.loadtxt("Dis_roc_pr/MDASAEtpr.txt"),
    'LAGCN': np.loadtxt("Dis_roc_pr/LAGCNy_ROC.txt"),
    'NNGCFCAE': np.loadtxt("Dis_roc_pr/tpr0_best.txt")
}
auc_d = {name: auc(fpr_d[name], tpr_d[name]) for name in fpr_d}

# ————— 绘图：一行两列子图 —————
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：HMDAD
for name in fpr_h:
    ax1.plot(fpr_h[name], tpr_h[name], label=f'{name} (AUC={auc_h[name]:.4f})')
ax1.set_title("ROC curves of HMDAD")
ax1.set_xlabel("FPR")
ax1.set_ylabel("TPR")
ax1.legend(loc='lower right')
ax1.grid(True)

# 右图：Disbiome
for name in fpr_d:
    ax2.plot(fpr_d[name], tpr_d[name], label=f'{name} (AUC={auc_d[name]:.4f})')
ax2.set_title("ROC curves of Disbiome")
ax2.set_xlabel("FPR")
ax2.set_ylabel("TPR")
ax2.legend(loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.show()
