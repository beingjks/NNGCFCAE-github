
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# phi 和 gamma 数值
phi_vals = [1, 10, 100, 1000]
gamma_vals = [1, 10, 100, 1000]

# 索引坐标
phi_pos = np.arange(len(phi_vals))
gamma_pos = np.arange(len(gamma_vals))
xx, yy = np.meshgrid(phi_pos, gamma_pos)
x = xx.ravel()
y = yy.ravel()
dx = dy = 0.5

# 原始数据
auc = np.array([
    [0.9803, 0.9864, 0.9773, 0.9775],
    [0.9804, 0.9762, 0.9773, 0.9774],
    [0.9784, 0.9773, 0.9774, 0.9775],
    [0.9783, 0.9772, 0.9776, 0.9766]
])

aupr = np.array([
    [0.9757, 0.9843, 0.9740, 0.9743],
    [0.9768, 0.9743, 0.9740, 0.9741],
    [0.9751, 0.9740, 0.9741, 0.9742],
    [0.9750, 0.9739, 0.9743, 0.9743]
])

# 设置每根柱子的底部起点（略低于数据最小值）
auc_min = auc.min()
aupr_min = aupr.min()
z_offset = 0.0001  # 防止柱子贴平

z_auc_base = np.full_like(auc.ravel(), auc_min - z_offset)
z_aupr_base = np.full_like(aupr.ravel(), aupr_min - z_offset)
auc_height = auc.ravel() - z_auc_base
aupr_height = aupr.ravel() - z_aupr_base

fig = plt.figure(figsize=(14, 6))

# -------- AUC 图 --------
ax1 = fig.add_subplot(121, projection='3d')
ax1.bar3d(x, y, z_auc_base, dx, dy, auc_height, color='skyblue', shade=True)
ax1.set_title("(a)AUC", fontsize=14)
ax1.set_xlabel(r"$\phi$")
ax1.set_ylabel(r"$\gamma$")
# ax1.set_zlabel("Score")
ax1.set_xticks(phi_pos + dx / 2)
ax1.set_xticklabels(phi_vals)
ax1.set_yticks(gamma_pos + dy / 2)
ax1.set_yticklabels(gamma_vals)
ax1.set_zlim(auc_min - z_offset, auc.max() + 0.001)
ax1.grid(False)  # ✅ 关闭网格线

# 原始值标签
for i in range(len(x)):
    ax1.text(x[i], y[i], auc.ravel()[i] + 0.0001, f"{auc.ravel()[i]:.4f}",
             ha='center', va='bottom', fontsize=8)

# -------- AUPR 图 --------
ax2 = fig.add_subplot(122, projection='3d')
ax2.bar3d(x, y, z_aupr_base, dx, dy, aupr_height, color='coral', shade=True)
ax2.set_title("(a)AUPR", fontsize=14)
ax2.set_xlabel(r"$\phi$")
ax2.set_ylabel(r"$\gamma$")
# ax2.set_zlabel("Score")
ax2.set_xticks(phi_pos + dx / 2)
ax2.set_xticklabels(phi_vals)
ax2.set_yticks(gamma_pos + dy / 2)
ax2.set_yticklabels(gamma_vals)
ax2.set_zlim(aupr_min - z_offset, aupr.max() + 0.001)
ax2.grid(False)  # ✅ 关闭网格线

# 原始值标签
for i in range(len(x)):
    ax2.text(x[i], y[i], aupr.ravel()[i] + 0.0001, f"{aupr.ravel()[i]:.4f}",
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # φ 和 γ 数值
# phi_vals = [1, 10, 100, 1000]
# gamma_vals = [1, 10, 100, 1000]
# phi_pos = np.arange(len(phi_vals))
# gamma_pos = np.arange(len(gamma_vals))
# xx, yy = np.meshgrid(phi_pos, gamma_pos)
# x = xx.ravel()
# y = yy.ravel()
# dx = dy = 0.5
#
# # 新提取的数据
# auc = np.array([
#     [0.9601, 0.9598, 0.9599, 0.9601],
#     [0.9599, 0.9598, 0.9598, 0.9601],
#     [0.9600, 0.9598, 0.9599, 0.9602],
#     [0.9599, 0.9598, 0.9600, 0.9603]
# ])
# aupr = np.array([
#     [0.9582, 0.9578, 0.9579, 0.9580],
#     [0.9579, 0.9578, 0.9578, 0.9581],
#     [0.9580, 0.9577, 0.9578, 0.9582],
#     [0.9579, 0.9578, 0.9580, 0.9582]
# ])
#
# # 柱状图从最小值开始，不穿 XY 平面
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
# ax1.bar3d(x, y, z_auc_base, dx, dy, auc_height, color='steelblue', shade=True)
# ax1.set_title("(b)AUC", fontsize=14)
# ax1.set_xlabel(r"$\phi$")
# ax1.set_ylabel(r"$\gamma$")
# # ax1.set_zlabel("Score")
# ax1.set_xticks(phi_pos + dx / 2)
# ax1.set_xticklabels(phi_vals)
# ax1.set_yticks(gamma_pos + dy / 2)
# ax1.set_yticklabels(gamma_vals)
# ax1.set_zlim(auc_min - z_offset, auc.max() + 0.0003)
# ax1.grid(False)
#
# # 数据标签
# for i in range(len(x)):
#     ax1.text(x[i], y[i], auc.ravel()[i] + 0.00005, f"{auc.ravel()[i]:.4f}",
#              ha='center', va='bottom', fontsize=8)
#
# # -------- AUPR 图 --------
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.bar3d(x, y, z_aupr_base, dx, dy, aupr_height, color='indianred', shade=True)
# ax2.set_title("(b)AUPR", fontsize=14)
# ax2.set_xlabel(r"$\phi$")
# ax2.set_ylabel(r"$\gamma$")
# # ax2.set_zlabel("Score")
# ax2.set_xticks(phi_pos + dx / 2)
# ax2.set_xticklabels(phi_vals)
# ax2.set_yticks(gamma_pos + dy / 2)
# ax2.set_yticklabels(gamma_vals)
# ax2.set_zlim(aupr_min - z_offset, aupr.max() + 0.0003)
# ax2.grid(False)
#
# for i in range(len(x)):
#     ax2.text(x[i], y[i], aupr.ravel()[i] + 0.00005, f"{aupr.ravel()[i]:.4f}",
#              ha='center', va='bottom', fontsize=8)
#
# plt.tight_layout()
# plt.show()
