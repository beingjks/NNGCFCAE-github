# 图1 lrl
import matplotlib.pyplot as plt

# # 示例数据
# lr = [0.001, 0.01, 0.05, 0.1]
# auc = [0.9743, 0.9770, 0.9768, 0.9769]

# 绘图
# plt.plot(lr, auc, marker='o', color='orange')  # 折线图
# plt.xlabel('lr')  # x轴标签
# plt.ylabel('AUC Value')  # y轴标签
# plt.title('AUC vs Learning Rate')  # 图标题
# plt.grid(True)
# plt.show()

import matplotlib.pyplot as plt

# # 示例数据
# lr = [0.001, 0.01, 0.05, 0.1]
# auc = [0.9750, 0.9771, 0.9764, 0.9769]
#
# # 绘图
# plt.plot(lr, auc, marker='o', color='blue')  # 蓝色折线图
# plt.xlabel('lr')  # x轴标签
# plt.ylabel('AUC')  # y轴标签
# #plt.title('AUC vs Learning Rate 1')  # 图标题
# plt.show()

#
# import matplotlib.pyplot as plt
# import numpy as np
#

import matplotlib.pyplot as plt
#
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
#
# # 学习率 (lr2) 和对应的 AUC 值
# lr_values = [0.001, 0.01, 0.05, 0.1]
# auc_values = [0.9722, 0.9735, 0.9769, 0.9771]
#
# # 创建图形
# plt.figure(figsize=(8, 5))
#
# # 绘制蓝色折线图（无数据标签）
# plt.plot(lr_values, auc_values, color='blue', marker='o')
#
# # 设置标题和坐标轴标签
# plt.xlabel("lr")
# plt.ylabel("AUC value")
# plt.title('AUC vs Learning Rate 2')  # 图标题
# # 设置 y 轴格式为 4 位小数
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
#
# # 自动布局
# plt.tight_layout()
#
# # 显示图形
# plt.show()
#


#
# import matplotlib.pyplot as plt
#
# # 横坐标标签（ω-γ 比例）
# x_labels = ['1-1', '1-10', '1-100', '1-1000',
#             '10-1', '10-10', '10-100', '10-1000',
#             '100-1', '100-10', '100-100', '1000-1', '1000-10', '1000-100', '1000-1000']
#
# # AUC 值
# auc_values = [0.9769, 0.9719,0.9759 , 0.9725,
#               0.9771, 0.9703, 0.9704, 0.9718,
#               0.9741, 0.9713, 0.9740, 0.9715,
#               0.9724, 0.9760, 0.9721]
#
# # 创建图形
# plt.figure(figsize=(12, 6))
# plt.plot(x_labels, auc_values, marker='o', color='blue')  # 蓝色折线图，实心圆点
#
# # 添加数值标签
# for i, value in enumerate(auc_values):
#     plt.text(i, value + 0.001, f"{value:.4f}", ha='center', va='bottom', fontsize=9)
#
# # 图形标题和坐标轴标签
# plt.title("The AUC values on different ω-γ", fontsize=14)
# plt.ylabel("AUC")
#
# # 设置 y 轴范围
# plt.ylim(0.96, 1.00)
#
#
#
# # 横坐标角度倾斜
# plt.xticks(rotation=45)
#
# # 自动调整布局
# plt.tight_layout()
#
# # 显示图形
# plt.show()


#k2
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
#
# # k 值和对应的 AUC 值
# k_values = [32, 64, 128, 256]
# auc_values = [0.9743, 0.9765, 0.9770, 0.9743]
#
# # 创建图形
# plt.figure(figsize=(8, 5))
#
# # 绘制蓝色折线图（marker为圆点）
# plt.plot(k_values, auc_values, color='blue', marker='o')
#
# # 设置坐标轴标签
# plt.xlabel("k")
# plt.ylabel("AUC value")
# plt.title('AUC vs K2')  # 图标题
#
# # 设置 y 轴刻度为 4 位小数
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
#
# # 自动布局
# plt.tight_layout()
#
# # 显示图形
# plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
#
# # k 值 和 AUC 值（根据图像估计）
# k_values = [32, 64, 128, 256]
# auc_values = [0.9769, 0.9741, 0.9731, 0.9652]
#
# # 创建图形
# plt.figure(figsize=(8, 5))
#
# # 画蓝色折线图
# plt.plot(k_values, auc_values, color='blue', marker='o')
#
# # 设置坐标轴
# plt.xlabel("k1")
# plt.ylabel("AUC value")
# plt.title('AUC vs K1')  # 图标题
#
# # 设置 y 轴格式为四位小数
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
#
# # 自动布局
# plt.tight_layout()
#
# # 显示图形
# plt.show()
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
#
# # channel 值 和 AUC 值（根据图像估计）
# channel_values = [3, 6, 9]
# auc_values = [0.9723, 0.9771, 0.9751]
#
# # 创建图形
# plt.figure(figsize=(8, 5))
#
# # 绘制蓝色线条
# plt.plot(channel_values, auc_values, color='blue', marker='o')
#
# # 设置坐标轴标签
# plt.xlabel("channel C")
# plt.ylabel("AUC value")
# plt.title('AUC vs channel C')  # 图标题
#
# # 设置 Y 轴格式为四位小数
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
#
# # 自动布局
# plt.tight_layout()
#
# # 显示图形
# plt.show()
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 横坐标：对数刻度参数
# x = [0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100, 200]
#
# # α 和 β 对应的 AUC（示意值，根据图形估算）
# alpha_auc = [0.93, 0.92, 0.925, 0.945, 0.955, 0.965, 0.975, 0.965, 0.96]
# beta_auc  = [0.96, 0.961, 0.965, 0.965, 0.971, 0.965, 0.96, 0.955, 0.955]
#
# # 创建图像
# plt.figure(figsize=(6, 4))
#
# # 绘制 α 和 β 曲线
# plt.plot(x, alpha_auc, marker='^', color='gold', label='α')   # 黄色三角
# plt.plot(x, beta_auc, marker='*', color='blue', label='β')  # 紫色星号
#
# # 对数坐标轴
# plt.xscale('log')
#
# # 垂直虚线（x=1 和 x=50）
# plt.axvline(x=1, linestyle='--', color='blue')
# plt.axvline(x=50, linestyle='--', color='gold')
# plt.title('AUC vs α,β')  # 图标题
#
# # 轴标签和图例
# plt.ylabel("AUC")
# plt.xlabel("α,β")
#
# plt.legend()
#
# # 设置 y 轴范围（根据图像）
# plt.ylim(0.8, 1.0)
#
# # 显示图形
# plt.tight_layout()
# plt.show()



import matplotlib.pyplot as plt

# 示例数据
N = [0.2, 0.4, 0.5, 0.6,0.8,1.0]
auc = [0.9550, 0.9671, 0.9770, 0.9669, 0.9570, 0.9550]

# 绘图
plt.plot(N, auc, marker='o', color='blue')  # 蓝色折线图
plt.xlabel(r'$\eta$')
plt.ylabel('AUC')  # y轴标签
plt.title('AUC vs ' + r'$\eta$')
plt.show()
