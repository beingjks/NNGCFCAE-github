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
# plt.title("PR Curves of HMDAD")
# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.legend(loc='lower left')
# plt.grid(True)
# plt.show()
#
#
#
# import matplotlib.pyplot as plt
# from sklearn.metrics import auc
# import numpy as np
#
# # 假设以下为模型的 recall 和 precision 值（需要你用真实数据替换）
# recall_gsamda = np.loadtxt("Dis_roc_pr/recallGSAMDA.txt")
# precision_gsamda = np.loadtxt("Dis_roc_pr/precisionGSAMDA.txt")
#
# recall_mdasae = np.loadtxt("Dis_roc_pr/MDASAEre.txt")
# precision_mdasae = np.loadtxt("Dis_roc_pr/MDASAEpr.txt")
#
#
# recall_lagcn = np.loadtxt("Dis_roc_pr/LAGCNx_PR.txt")
# precision_lagcn = np.loadtxt("Dis_roc_pr/LAGCNy_PR.txt")
#
# recall_gvnnvae = np.loadtxt("Dis_roc_pr/AB_GVNNVAE_recall0.txt")
# precision_gvnnvae = np.loadtxt("Dis_roc_pr/AB_GVNNVAE_precision0.txt")
#
# recall_gcnmda = np.loadtxt("Dis_roc_pr/GCNMDArecall.txt")
# precision_gcnmda = np.loadtxt("Dis_roc_pr/GCNMDAprecision.txt")
# #
# # recall_nngcfcae = np.loadtxt("Dis_roc_pr/recall9.txt")
# # precision_nngcfcae = np.loadtxt("Dis_roc_pr/precision9.txt")
# # recall_nngcfcae = np.loadtxt("Dis_roc_pr/recall0_best.txt")
# # precision_nngcfcae = np.loadtxt("Dis_roc_pr/precision0_best.txt")
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
# plt.title("PR Curves of Disbiome")
# plt.xlabel("recall")
# plt.ylabel("precision")
# plt.legend(loc='lower left')
# plt.grid(True)
# plt.show()

#
#
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import auc
#
# # ————— 加载并计算 HMDAD 的 recall/precision & AUPR —————
# models_h = {
#     'MDASAE': ('HMDAD_roc_pr/MDASAEre.txt',    'HMDAD_roc_pr/MDASAEpr.txt'),
#     'GSAMDA': ('HMDAD_roc_pr/GSAMDArecall.txt','HMDAD_roc_pr/GSAMDAprecision.txt'),
#     'GVNNSAE':('HMDAD_roc_pr/MDAD_GVNNSAE_recall.txt','HMDAD_roc_pr/MDAD_GVNNSAE_precision.txt'),
#     'GCNMDA': ('HMDAD_roc_pr/GCNMDArecall.txt','HMDAD_roc_pr/GCNMDAprecision.txt'),
#     'LAGCN':  ('HMDAD_roc_pr/LAGCNx_PR00.txt','HMDAD_roc_pr/LAGCNy_PR00.txt'),
#     'NNGCFCAE':('HMDAD_roc_pr/recall0_best2.txt','HMDAD_roc_pr/precision0_best2.txt'),
# }
# data_h = {}
# aupr_h = {}
# for name, (rec_path, pr_path) in models_h.items():
#     rec = np.loadtxt(rec_path)
#     pr  = np.loadtxt(pr_path)
#     data_h[name] = (rec, pr)
#     aupr_h[name] = auc(rec, pr)
#
# # ————— 加载并计算 Disbiome 的 recall/precision & AUPR —————
# models_d = {
#     'MDASAE': ('Dis_roc_pr/MDASAEre.txt',    'Dis_roc_pr/MDASAEpr.txt'),
#     'GSAMDA':('Dis_roc_pr/recallGSAMDA.txt','Dis_roc_pr/precisionGSAMDA.txt'),
#     'GVNNSAE':('Dis_roc_pr/AB_GVNNVAE_recall0.txt','Dis_roc_pr/AB_GVNNVAE_precision0.txt'),
#     'GCNMDA': ('Dis_roc_pr/GCNMDArecall.txt','Dis_roc_pr/GCNMDAprecision.txt'),
#     'LAGCN':  ('Dis_roc_pr/LAGCNx_PR.txt','Dis_roc_pr/LAGCNy_PR.txt'),
#     'NNGCFCAE':('Dis_roc_pr/recall0_best.txt','Dis_roc_pr/precision0_best.txt'),
# }
# data_d = {}
# aupr_d = {}
# for name, (rec_path, pr_path) in models_d.items():
#     rec = np.loadtxt(rec_path)
#     pr  = np.loadtxt(pr_path)
#     data_d[name] = (rec, pr)
#     aupr_d[name] = auc(rec, pr)
#
# # ————— 绘图：一行两列子图 —————
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#
# # 左：HMDAD
# for name, (rec, pr) in data_h.items():
#     ax1.plot(rec, pr, label=f'{name} (AUPR={aupr_h[name]:.4f})')
# ax1.set_title("PR Curves of HMDAD")
# ax1.set_xlabel("Recall")
# ax1.set_ylabel("Precision")
# ax1.legend(loc='lower left')
# ax1.grid(True)
#
# # 右：Disbiome
# for name, (rec, pr) in data_d.items():
#     ax2.plot(rec, pr, label=f'{name} (AUPR={aupr_d[name]:.4f})')
# ax2.set_title("PR Curves of Disbiome")
# ax2.set_xlabel("Recall")
# ax2.set_ylabel("Precision")
# ax2.legend(loc='lower left')
# ax2.grid(True)
#
# plt.tight_layout()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# ————— 加载 HMDAD —————
recall_gsamda_h = np.loadtxt("HMDAD_roc_pr/GSAMDArecall.txt")
precision_gsamda_h = np.loadtxt("HMDAD_roc_pr/GSAMDAprecision.txt")
recall_mdasae_h = np.loadtxt("HMDAD_roc_pr/MDASAEre.txt")
precision_mdasae_h = np.loadtxt("HMDAD_roc_pr/MDASAEpr.txt")
recall_lagcn_h = np.loadtxt("HMDAD_roc_pr/LAGCNx_PR00.txt")
precision_lagcn_h = np.loadtxt("HMDAD_roc_pr/LAGCNy_PR00.txt")
recall_gvnnvae_h = np.loadtxt("HMDAD_roc_pr/MDAD_GVNNSAE_recall.txt")
precision_gvnnvae_h = np.loadtxt("HMDAD_roc_pr/MDAD_GVNNSAE_precision.txt")
recall_gcnmda_h = np.loadtxt("HMDAD_roc_pr/GCNMDArecall.txt")
precision_gcnmda_h = np.loadtxt("HMDAD_roc_pr/GCNMDAprecision.txt")
recall_nngcfcae_h = np.loadtxt("HMDAD_roc_pr/recall0_best2.txt")
precision_nngcfcae_h = np.loadtxt("HMDAD_roc_pr/precision0_best2.txt")

# ————— 加载 Disbiome —————
recall_gsamda_d = np.loadtxt("Dis_roc_pr/recallGSAMDA.txt")
precision_gsamda_d = np.loadtxt("Dis_roc_pr/precisionGSAMDA.txt")
recall_mdasae_d = np.loadtxt("Dis_roc_pr/MDASAEre.txt")
precision_mdasae_d = np.loadtxt("Dis_roc_pr/MDASAEpr.txt")
recall_lagcn_d = np.loadtxt("Dis_roc_pr/LAGCNx_PR.txt")
precision_lagcn_d = np.loadtxt("Dis_roc_pr/LAGCNy_PR.txt")
recall_gvnnvae_d = np.loadtxt("Dis_roc_pr/AB_GVNNVAE_recall0.txt")
precision_gvnnvae_d = np.loadtxt("Dis_roc_pr/AB_GVNNVAE_precision0.txt")
recall_gcnmda_d = np.loadtxt("Dis_roc_pr/GCNMDArecall.txt")
precision_gcnmda_d = np.loadtxt("Dis_roc_pr/GCNMDAprecision.txt")
recall_nngcfcae_d = np.loadtxt("Dis_roc_pr/recall0_best.txt")
precision_nngcfcae_d = np.loadtxt("Dis_roc_pr/precision0_best.txt")

# ————— 计算 AUPR —————
def calc_aupr(r, p): return auc(r, p)

# ————— 颜色映射（保持你原来的颜色顺序）—————
colors = {
    'MDASAE': 'darkorange',
    'GSAMDA': 'green',
    'GVNNVAE': 'mediumpurple',
    'LAGCN': 'brown',
    'GCNMDA': 'crimson',
    'NNGCFCAE': 'cornflowerblue',
}

# ————— 绘图 —————
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图 HMDAD
ax1.plot(recall_mdasae_h, precision_mdasae_h, label=f'MDASAE (AUPR={calc_aupr(recall_mdasae_h, precision_mdasae_h):.4f})', color=colors['MDASAE'])
ax1.plot(recall_gsamda_h, precision_gsamda_h, label=f'GSAMDA (AUPR={calc_aupr(recall_gsamda_h, precision_gsamda_h):.4f})', color=colors['GSAMDA'])
ax1.plot(recall_gvnnvae_h, precision_gvnnvae_h, label=f'GVNNVAE (AUPR={calc_aupr(recall_gvnnvae_h, precision_gvnnvae_h):.4f})', color=colors['GVNNVAE'])
ax1.plot(recall_lagcn_h, precision_lagcn_h, label=f'LAGCN (AUPR={calc_aupr(recall_lagcn_h, precision_lagcn_h):.4f})', color=colors['LAGCN'])
ax1.plot(recall_gcnmda_h, precision_gcnmda_h, label=f'GCNMDA (AUPR={calc_aupr(recall_gcnmda_h, precision_gcnmda_h):.4f})', color=colors['GCNMDA'])
ax1.plot(recall_nngcfcae_h, precision_nngcfcae_h, label=f'NNGCFCAE (AUPR={calc_aupr(recall_nngcfcae_h, precision_nngcfcae_h):.4f})', color=colors['NNGCFCAE'])
ax1.set_title("PR Curves of HMDAD")
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.legend(loc='lower left')
ax1.grid(True)

# 右图 Disbiome
ax2.plot(recall_mdasae_d, precision_mdasae_d, label=f'MDASAE (AUPR={calc_aupr(recall_mdasae_d, precision_mdasae_d):.4f})', color=colors['MDASAE'])
ax2.plot(recall_gsamda_d, precision_gsamda_d, label=f'GSAMDA (AUPR={calc_aupr(recall_gsamda_d, precision_gsamda_d):.4f})', color=colors['GSAMDA'])
ax2.plot(recall_gvnnvae_d, precision_gvnnvae_d, label=f'GVNNVAE (AUPR={calc_aupr(recall_gvnnvae_d, precision_gvnnvae_d):.4f})', color=colors['GVNNVAE'])
ax2.plot(recall_lagcn_d, precision_lagcn_d, label=f'LAGCN (AUPR={calc_aupr(recall_lagcn_d, precision_lagcn_d):.4f})', color=colors['LAGCN'])
ax2.plot(recall_gcnmda_d, precision_gcnmda_d, label=f'GCNMDA (AUPR={calc_aupr(recall_gcnmda_d, precision_gcnmda_d):.4f})', color=colors['GCNMDA'])
ax2.plot(recall_nngcfcae_d, precision_nngcfcae_d, label=f'NNGCFCAE (AUPR={calc_aupr(recall_nngcfcae_d, precision_nngcfcae_d):.4f})', color=colors['NNGCFCAE'])
ax2.set_title("PR Curves of Disbiome")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc='lower left')
ax2.grid(True)

plt.tight_layout()
plt.show()
