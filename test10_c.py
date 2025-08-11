# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ===============================
# # 1. 数据读取
# # ===============================
# c_values = ['3', '6', '9']   # 文件夹名称
# base_path = r'C:\Users\胡先俊\Desktop\论文\实验数据\超参数分析\c'  # 路径不变
#
# auc_data = []
# aupr_data = []
#
# for c in c_values:
#     auc_file = os.path.join(base_path, c, 'AUC--------CSAEtt', 'auc.txt')
#     aupr_file = os.path.join(base_path, c, 'AUC--------CSAEtt', 'aupr.txt')
#
#     # 读取 AUC
#     if os.path.exists(auc_file):
#         with open(auc_file, 'r', encoding='utf-8') as f:
#             auc_data.append([float(line.strip()) for line in f if line.strip()])
#     else:
#         print(f'❌ 没找到文件: {auc_file}')
#         auc_data.append([])
#
#     # 读取 AUPR
#     if os.path.exists(aupr_file):
#         with open(aupr_file, 'r', encoding='utf-8') as f:
#             aupr_data.append([float(line.strip()) for line in f if line.strip()])
#     else:
#         print(f'❌ 没找到文件: {aupr_file}')
#         aupr_data.append([])
#
# # 转成 numpy
# auc_data = [np.array(x) for x in auc_data]
# aupr_data = [np.array(x) for x in aupr_data]
#
# # ===============================
# # 2. 绘制箱线图
# # ===============================
# x = np.arange(len(c_values))
# fig, ax1 = plt.subplots(figsize=(9,5))
#
# box_width = 0.15
# offset = 0.12
#
# # ---- AUC (左轴) ----
# bp_auc = ax1.boxplot(auc_data, positions=x-offset, widths=box_width,
#                      patch_artist=True, showfliers=False,
#                      boxprops=dict(facecolor='lightblue', color='blue'),
#                      medianprops=dict(color='blue', linewidth=1.5),
#                      whiskerprops=dict(color='blue'),
#                      capprops=dict(color='blue'))
# ax1.set_ylabel('AUROC', color='blue', fontsize=12)
# ax1.tick_params(axis='y', labelcolor='blue')
#
# # 标注 AUC 中位数
# for i, vals in enumerate(auc_data):
#     if len(vals) > 0:
#         med = np.median(vals)
#         ax1.text(i-offset-0.08, med, f'{med:.2f}', color='blue',
#                  fontsize=10, ha='right', va='center', fontweight='bold')
#
# # ---- AUPR (右轴) ----
# ax2 = ax1.twinx()
# bp_aupr = ax2.boxplot(aupr_data, positions=x+offset, widths=box_width,
#                       patch_artist=True, showfliers=False,
#                       boxprops=dict(facecolor='mistyrose', color='red'),
#                       medianprops=dict(color='red', linewidth=1.5),
#                       whiskerprops=dict(color='red'),
#                       capprops=dict(color='red'))
# ax2.set_ylabel('AUPR', color='red', fontsize=12)
# ax2.tick_params(axis='y', labelcolor='red')
#
# # 标注 AUPR 中位数
# for i, vals in enumerate(aupr_data):
#     if len(vals) > 0:
#         med = np.median(vals)
#         ax2.text(i+offset+0.08, med, f'{med:.2f}', color='red',
#                  fontsize=10, ha='left', va='center', fontweight='bold')
#
# # ---- X轴 ----
# ax1.set_xticks(x)
# ax1.set_xticklabels(c_values, fontsize=11)
# ax1.set_xlabel('channel C', fontsize=12, labelpad=10)  # ✅ x轴标签为 c
#
# # ---- 图例（左下角） ----
# legend_handles = [bp_auc['boxes'][0], bp_aupr['boxes'][0]]
# legend_labels = ['AUROC', 'AUPR']
# ax1.legend(legend_handles, legend_labels, fontsize=12,
#            loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=True)
#
#
# # ---- 标题 ----
# plt.title('Channel Parameter (C) ', fontsize=14, fontweight='bold')
#
# # ---- 美化 ----
# ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
# ax1.set_facecolor('whitesmoke')
# plt.tight_layout()
# plt.show()


#
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ===============================
# # 1. 数据读取
# # ===============================
# c_values = ['3', '6', '9']   # 文件夹名称
# base_path = r'C:\Users\胡先俊\Desktop\论文\实验数据\超参数分析\c'
#
# auc_data = []
# aupr_data = []
#
# for c in c_values:
#     auc_file = os.path.join(base_path, c, 'AUC--------CSAEtt', 'auc.txt')
#     aupr_file = os.path.join(base_path, c, 'AUC--------CSAEtt', 'aupr.txt')
#
#     # 读取 AUC
#     if os.path.exists(auc_file):
#         with open(auc_file, 'r', encoding='utf-8') as f:
#             auc_data.append([float(line.strip()) for line in f if line.strip()])
#     else:
#         print(f'❌ 没找到文件: {auc_file}')
#         auc_data.append([])
#
#     # 读取 AUPR
#     if os.path.exists(aupr_file):
#         with open(aupr_file, 'r', encoding='utf-8') as f:
#             aupr_data.append([float(line.strip()) for line in f if line.strip()])
#     else:
#         print(f'❌ 没找到文件: {aupr_file}')
#         aupr_data.append([])
#
# # 转成 numpy
# auc_data = [np.array(x) for x in auc_data]
# aupr_data = [np.array(x) for x in aupr_data]
#
# # ===============================
# # 2. 绘制箱线图
# # ===============================
# x = np.arange(len(c_values))
# fig, ax1 = plt.subplots(figsize=(9,5))
#
# box_width = 0.15
# offset    = 0.12
# line_w    = 2    # 箱线、须线、盖帽加粗线宽
# median_w  = 3    # 中位数线更粗
#
# # ---- AUC (左轴) ----
# bp_auc = ax1.boxplot(
#     auc_data,
#     positions=x-offset,
#     widths=box_width,
#     patch_artist=True,
#     showfliers=False,
#     boxprops=dict(facecolor='lightblue', edgecolor='navy', linewidth=line_w),
#     whiskerprops=dict(color='navy', linewidth=line_w),
#     capprops=dict(color='navy', linewidth=line_w),
#     medianprops=dict(color='navy', linewidth=median_w)
# )
# # ax1.set_ylabel('AUROC', color='navy', fontsize=12)
# ax1.tick_params(axis='y', labelcolor='navy')
#
# # 标注 AUC 中位数
# for i, vals in enumerate(auc_data):
#     if vals.size:
#         med = np.median(vals)
#         ax1.text(i-offset-0.08, med, f'{med:.2f}',
#                  color='navy', fontsize=10,
#                  ha='right', va='center', fontweight='bold')
#
# # ---- AUPR (右轴) ----
# ax2 = ax1.twinx()
# bp_aupr = ax2.boxplot(
#     aupr_data,
#     positions=x+offset,
#     widths=box_width,
#     patch_artist=True,
#     showfliers=False,
#     boxprops=dict(facecolor='mistyrose', edgecolor='darkred', linewidth=line_w),
#     whiskerprops=dict(color='darkred', linewidth=line_w),
#     capprops=dict(color='darkred', linewidth=line_w),
#     medianprops=dict(color='darkred', linewidth=median_w)
# )
# ax2.set_ylabel('AUPR', color='darkred', fontsize=12)
# ax2.tick_params(axis='y', labelcolor='darkred')
#
# # 标注 AUPR 中位数
# for i, vals in enumerate(aupr_data):
#     if vals.size:
#         med = np.median(vals)
#         ax2.text(i+offset+0.08, med, f'{med:.2f}',
#                  color='darkred', fontsize=10,
#                  ha='left', va='center', fontweight='bold')
#
# # ---- X轴 ----
# ax1.set_xticks(x)
# ax1.set_xticklabels(c_values, fontsize=11)
# ax1.set_xlabel('channel C', fontsize=12, labelpad=10)
#
# # ---- 图例（左上角） ----
# legend_handles = [bp_auc['boxes'][0], bp_aupr['boxes'][0]]
# legend_labels  = ['AUROC', 'AUPR']
# ax1.legend(legend_handles, legend_labels, fontsize=12,
#            loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=True)
#
# # ---- 标题 ----
# plt.title('Channel Parameter (C)', fontsize=14, fontweight='bold')
#
# # ---- 美化 ----
# ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
# ax1.set_facecolor('whitesmoke')
# plt.tight_layout()
# plt.show()







import os
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. 数据读取
# ===============================
c_values = ['3', '6', '9']   # 文件夹名称
base_path = r'C:\Users\胡先俊\Desktop\论文\实验数据\超参数分析\dis\c'

auc_data = []
aupr_data = []

for c in c_values:
    auc_file = os.path.join(base_path, c, 'AUC--------CSAEtt', 'auc.txt')
    aupr_file = os.path.join(base_path, c, 'AUC--------CSAEtt', 'aupr.txt')

    # 读取 AUC
    if os.path.exists(auc_file):
        with open(auc_file, 'r', encoding='utf-8') as f:
            auc_data.append([float(line.strip()) for line in f if line.strip()])
    else:
        print(f'❌ 没找到文件: {auc_file}')
        auc_data.append([])

    # 读取 AUPR
    if os.path.exists(aupr_file):
        with open(aupr_file, 'r', encoding='utf-8') as f:
            aupr_data.append([float(line.strip()) for line in f if line.strip()])
    else:
        print(f'❌ 没找到文件: {aupr_file}')
        aupr_data.append([])

# 转成 numpy
auc_data = [np.array(x) for x in auc_data]
aupr_data = [np.array(x) for x in aupr_data]

# ===============================
# 2. 绘制箱线图
# ===============================
x = np.arange(len(c_values))
fig, ax1 = plt.subplots(figsize=(9,5))

box_width = 0.15
offset    = 0.12
line_w    = 2    # 箱线、须线、盖帽加粗线宽
median_w  = 3    # 中位数线更粗

# ---- AUC (左轴) ----
bp_auc = ax1.boxplot(
    auc_data,
    positions=x-offset,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor='lightblue', edgecolor='navy', linewidth=line_w),
    whiskerprops=dict(color='navy', linewidth=line_w),
    capprops=dict(color='navy', linewidth=line_w),
    medianprops=dict(color='navy', linewidth=median_w)
)
# ax1.set_ylabel('AUROC', color='navy', fontsize=12)
ax1.tick_params(axis='y', labelcolor='navy')

# 标注 AUC 中位数
for i, vals in enumerate(auc_data):
    if vals.size:
        med = np.median(vals)
        ax1.text(i-offset-0.08, med, f'{med:.2f}',
                 color='navy', fontsize=10,
                 ha='right', va='center', fontweight='bold')

# ---- AUPR (右轴) ----
ax2 = ax1.twinx()
bp_aupr = ax2.boxplot(
    aupr_data,
    positions=x+offset,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor='mistyrose', edgecolor='darkred', linewidth=line_w),
    whiskerprops=dict(color='darkred', linewidth=line_w),
    capprops=dict(color='darkred', linewidth=line_w),
    medianprops=dict(color='darkred', linewidth=median_w)
)
ax2.set_ylabel('AUPR', color='darkred', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkred')

# 标注 AUPR 中位数
for i, vals in enumerate(aupr_data):
    if vals.size:
        med = np.median(vals)
        ax2.text(i+offset+0.08, med, f'{med:.2f}',
                 color='darkred', fontsize=10,
                 ha='left', va='center', fontweight='bold')

# ---- X轴 ----
ax1.set_xticks(x)
ax1.set_xticklabels(c_values, fontsize=11)
ax1.set_xlabel('channel C', fontsize=12, labelpad=10)

# ---- 图例（左上角） ----
legend_handles = [bp_auc['boxes'][0], bp_aupr['boxes'][0]]
legend_labels  = ['AUROC', 'AUPR']
ax1.legend(legend_handles, legend_labels, fontsize=12,
           loc='lower right', bbox_to_anchor=(0.98, 0.02), frameon=True)

# ---- 标题 ----
plt.title('Channel Parameter (C)', fontsize=14, fontweight='bold')

# ---- 美化 ----
ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
ax1.set_facecolor('whitesmoke')
plt.tight_layout()
plt.show()
