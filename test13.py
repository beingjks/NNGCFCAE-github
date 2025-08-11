# import numpy as np
# import matplotlib.pyplot as plt
#
# # 参数设置
# alpha_vals = [0.01, 1, 10, 50]
# beta_vals = [0.01, 1, 10, 50]
# alpha_pos = np.arange(len(alpha_vals))
# beta_pos = np.arange(len(beta_vals))
# xx, yy = np.meshgrid(alpha_pos, beta_pos)
# x = xx.ravel()
# y = yy.ravel()
# dx = dy = 0.5
#
# # HMDAD 表格中的数据
# auc = np.array([
#     [0.9789, 0.9775, 0.9783, 0.9778],
#     [0.9778, 0.9778, 0.9781, 0.9778],
#     [0.9773, 0.9780, 0.9781, 0.9779],
#     [0.9772, 0.9782, 0.9780, 0.9779]
# ])
# aupr = np.array([
#     [0.9760, 0.9742, 0.9751, 0.9746],
#     [0.9746, 0.9746, 0.9748, 0.9746],
#     [0.9740, 0.9747, 0.9748, 0.9746],
#     [0.9739, 0.9750, 0.9748, 0.9747]
# ])
#
# # 柱子起点处理
# auc_min = auc.min()
# aupr_min = aupr.min()
# z_offset = 0.00005
#
# z_auc_base = np.full_like(auc.ravel(), auc_min - z_offset)
# z_aupr_base = np.full_like(aupr.ravel(), aupr_min - z_offset)
# auc_height = auc.ravel() - z_auc_base
# aupr_height = aupr.ravel() - z_aupr_base
#
# fig = plt.figure(figsize=(14, 6))
#
# # -------- AUC 图 --------
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.bar3d(x, y, z_auc_base, dx, dy, auc_height, color='skyblue', shade=True)
# ax1.set_title("(a)AUC", fontsize=14)
# ax1.set_xlabel(r"$\alpha$")
# ax1.set_ylabel(r"$\beta$")
# # ax1.set_zlabel("Score")
# ax1.set_xticks(alpha_pos + dx / 2)
# ax1.set_xticklabels(alpha_vals)
# ax1.set_yticks(beta_pos + dy / 2)
# ax1.set_yticklabels(beta_vals)
# ax1.set_zlim(auc_min - z_offset, auc.max() + 0.0003)
# ax1.grid(False)
# # ✅ 设置 Z 轴显示格式为四位小数
# from matplotlib.ticker import FormatStrFormatter
# ax1.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# for i in range(len(x)):
#     ax1.text(x[i], y[i], auc.ravel()[i] + 0.00005, f"{auc.ravel()[i]:.4f}",
#              ha='center', va='bottom', fontsize=8)
#
# # -------- AUPR 图 --------
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.bar3d(x, y, z_aupr_base, dx, dy, aupr_height, color='coral', shade=True)
# ax2.set_title("(a)AUPR ", fontsize=14)
# ax2.set_xlabel(r"$\alpha$")
# ax2.set_ylabel(r"$\beta$")
# # ax2.set_zlabel("Score")
# ax2.set_xticks(alpha_pos + dx / 2)
# ax2.set_xticklabels(alpha_vals)
# ax2.set_yticks(beta_pos + dy / 2)
# ax2.set_yticklabels(beta_vals)
# ax2.set_zlim(aupr_min - z_offset, aupr.max() + 0.0003)
# ax2.grid(False)
#
# for i in range(len(x)):
#     ax2.text(x[i], y[i], aupr.ravel()[i] + 0.00005, f"{aupr.ravel()[i]:.4f}",
#              ha='center', va='bottom', fontsize=8)
#
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# 参数设置
alpha_vals = [0.01, 1, 10, 50]
beta_vals = [0.01, 1, 10, 50]
alpha_pos = np.arange(len(alpha_vals))
beta_pos = np.arange(len(beta_vals))
xx, yy = np.meshgrid(alpha_pos, beta_pos)
x = xx.ravel()
y = yy.ravel()
dx = dy = 0.5

# Disbiome α-β 数据
auc = np.array([
    [0.9635, 0.9608, 0.9609, 0.9607],
    [0.9612, 0.9607, 0.9608, 0.9607],
    [0.9608, 0.9607, 0.9607, 0.9607],
    [0.9607, 0.9608, 0.9607, 0.9606]
])
aupr = np.array([
    [0.9611, 0.9585, 0.9586, 0.9584],
    [0.9588, 0.9584, 0.9585, 0.9585],
    [0.9585, 0.9584, 0.9585, 0.9585],
    [0.9585, 0.9586, 0.9584, 0.9584]
])

# 柱子起点处理
auc_min = auc.min()
aupr_min = aupr.min()
z_offset = 0.00005
z_auc_base = np.full_like(auc.ravel(), auc_min - z_offset)
z_aupr_base = np.full_like(aupr.ravel(), aupr_min - z_offset)
auc_height = auc.ravel() - z_auc_base
aupr_height = aupr.ravel() - z_aupr_base

fig = plt.figure(figsize=(14, 6))

# -------- AUC 图 --------
ax1 = fig.add_subplot(121, projection='3d')
ax1.bar3d(x, y, z_auc_base, dx, dy, auc_height, color='steelblue', shade=True)
ax1.set_title("(b)AUC", fontsize=14)
ax1.set_xlabel(r"$\alpha$")
ax1.set_ylabel(r"$\beta$")
ax1.set_xticks(alpha_pos + dx / 2)
ax1.set_xticklabels(alpha_vals)
ax1.set_yticks(beta_pos + dy / 2)
ax1.set_yticklabels(beta_vals)
ax1.set_zlim(auc_min - z_offset, auc.max() + 0.0003)
ax1.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax1.grid(False)

for i in range(len(x)):
    ax1.text(x[i], y[i], auc.ravel()[i] + 0.00005, f"{auc.ravel()[i]:.4f}",
             ha='center', va='bottom', fontsize=8)

# -------- AUPR 图 --------
ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d(x, y, z_aupr_base, dx, dy, aupr_height, color='indianred', shade=True)
ax2.set_title("(b)AUPR", fontsize=14)
ax2.set_xlabel(r"$\alpha$")
ax2.set_ylabel(r"$\beta$")
ax2.set_xticks(alpha_pos + dx / 2)
ax2.set_xticklabels(alpha_vals)
ax2.set_yticks(beta_pos + dy / 2)
ax2.set_yticklabels(beta_vals)
ax2.set_zlim(aupr_min - z_offset, aupr.max() + 0.0003)
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
ax2.grid(False)

for i in range(len(x)):
    ax2.text(x[i], y[i], aupr.ravel()[i] + 0.00005, f"{aupr.ravel()[i]:.4f}",
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
