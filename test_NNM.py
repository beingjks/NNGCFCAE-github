# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # # 读取三个迭代的矩阵数据
# # matrix_20 = np.loadtxt("A1_20.txt")
# # matrix_50 = np.loadtxt("A1_50.txt")
# # matrix_100 = np.loadtxt("A1_100.txt")
# #
# # # 截取每个矩阵的局部区域（例如左上角 30x30）
# # sub_20 = matrix_20[:30, :30]
# # sub_50 = matrix_50[:30, :30]
# # sub_100 = matrix_100[:30, :30]
# #
# # # 统一色阶范围
# # vmin = min(sub_20.min(), sub_50.min(), sub_100.min())
# # vmax = max(sub_20.max(), sub_50.max(), sub_100.max())
# #
# # # 设置绘图风格
# # plt.figure(figsize=(20, 6))
# #
# # # 绘制第20次迭代热图
# # plt.subplot(1, 3, 1)
# # sns.heatmap(sub_20, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
# # plt.title("Iteration 20", fontsize=14)
# #
# # # 绘制第50次迭代热图
# # plt.subplot(1, 3, 2)
# # sns.heatmap(sub_50, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
# # plt.title("Iteration 50", fontsize=14)
# #
# # # 绘制第100次迭代热图
# # plt.subplot(1, 3, 3)
# # sns.heatmap(sub_100, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
# # plt.title("Iteration 100", fontsize=14)
# #
# # # 布局优化
# # plt.tight_layout()
# # plt.show()
#
#
#
#
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 加载迭代矩阵数据（确保与文件名匹配）
# matrix_20 = np.loadtxt("A1_1000.txt")
# matrix_50 = np.loadtxt("A1_50.txt")
# matrix_100 = np.loadtxt("A1_100.txt")
#
# # 截取左上角子矩阵（30x30）
# # sub_20 = matrix_20[:30, :30]
# # sub_50 = matrix_50[:30, :30]
# # sub_100 = matrix_100[:30, :30]
#
#
# sub_20 = matrix_20[-30:, :30]
# sub_50 = matrix_50[-30:, :30]
# sub_100 = matrix_100[-30:, :30]
#
# # # 计算统一色阶的上下限（真实值，不做归一化）
# vmin = min(sub_20.min(), sub_50.min(), sub_100.min())
# vmax = max(sub_20.max(), sub_50.max(), sub_100.max())
#
# # 绘图
# plt.figure(figsize=(20, 6))
#
# plt.subplot(1, 3, 1)
# sns.heatmap(sub_20, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
# plt.title("Iteration 20", fontsize=14)
#
# plt.subplot(1, 3, 2)
# sns.heatmap(sub_50, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
# plt.title("Iteration 50", fontsize=14)
#
# plt.subplot(1, 3, 3)
# sns.heatmap(sub_100, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
# plt.title("Iteration 100", fontsize=14)
#
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载迭代矩阵数据（确保文件存在）
matrix_20 = np.loadtxt("A1_1000.txt")   # 对应的是第1000轮
matrix_100 = np.loadtxt("A1_100.txt")   # 第100轮

# 截取左下角区域（最后30行，前30列）
sub_20 = matrix_20[-30:, :30]
sub_100 = matrix_100[-30:, :30]

# 统一色阶
vmin = min(sub_20.min(), sub_100.min())
vmax = max(sub_20.max(), sub_100.max())

# 绘图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 2)
sns.heatmap(sub_20, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
plt.title("Iteration 1000", fontsize=14)

plt.subplot(1, 2, 1)
sns.heatmap(sub_100, cmap="YlGnBu", vmin=vmin, vmax=vmax, cbar=True)
plt.title("Iteration 100", fontsize=14)

plt.tight_layout()
plt.show()
