# # # import matplotlib.pyplot as plt
# # #
# # # # 模型名称与对应指标数据
# # # models = [
# # #     "NNM+GCF(w/o CAE)",
# # #     "NNM+CAE(w/o GCF)",
# # #     "GCF+CAE(w/o NNM)",
# # #     "NNM+GCF+CAE"
# # # ]
# # # auc = [0.9736, 0.9691, 0.9778, 0.9864]
# # # aupr = [0.9693, 0.9649, 0.9745, 0.9843]
# # #
# # # # 设置横轴位置与柱宽
# # # x = range(len(models))
# # # width = 0.35
# # #
# # # # 绘制柱状图
# # # fig, ax = plt.subplots(figsize=(10, 6))
# # # bars1 = ax.bar([i - width/2 for i in x], auc, width, label='AUC')
# # # bars2 = ax.bar([i + width/2 for i in x], aupr, width, label='AUPR')
# # #
# # # # 为每个柱子添加顶部数值标签
# # # for bar in bars1 + bars2:
# # #     yval = bar.get_height()
# # #     ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom')
# # #
# # # # 设置图表属性
# # # ax.set_xlabel('Models')
# # # ax.set_ylabel('Scores')
# # # ax.set_title('AUC and AUPR Comparison for HMDAD Dataset')
# # # ax.set_xticks(x)
# # # ax.set_xticklabels(models, rotation=15)
# # # ax.legend()
# # #
# # # # 自适应布局
# # # plt.tight_layout()
# # # plt.show()
# # #
# # #
# # #
# # # import matplotlib.pyplot as plt
# # #
# # # # 模型和对应的指标
# # # models = [
# # #     "NNM+GCF(w/o CAE)",
# # #     "NNM+CAE(w/o GCF)",
# # #     "GCF+CAE(w/o NNM)",
# # #     "NNM+GCF+CAE"
# # # ]
# # # auc = [0.9736, 0.9691, 0.9778, 0.9864]
# # # aupr = [0.9693, 0.9649, 0.9745, 0.9843]
# # #
# # # x = range(len(models))
# # # width = 0.35
# # #
# # # # 创建图表
# # # fig, ax = plt.subplots(figsize=(10, 6))
# # # bars1 = ax.bar([i - width/2 for i in x], auc, width, label='AUC', color='orange')
# # # bars2 = ax.bar([i + width/2 for i in x], aupr, width, label='AUPR', color='orangered')
# # #
# # # # 添加数据标签
# # # for bar in bars1 + bars2:
# # #     yval = bar.get_height()
# # #     ax.text(bar.get_x() + bar.get_width()/2, yval + 0.0005, f'{yval:.4f}', ha='center', va='bottom')
# # #
# # # # 设置Y轴范围以放大变化
# # # ax.set_ylim(0.96, 0.99)
# # #
# # # # 美化图表
# # # ax.set_xlabel('Models')
# # # # ax.set_ylabel('Scores')
# # # ax.set_title('AUC and AUPR Comparison for HMDAD Dataset ')
# # # ax.set_xticks(x)
# # # ax.set_xticklabels(models, rotation=15)
# # # ax.legend()
# # # plt.tight_layout()
# # #
# # # plt.show()
# # #
# # # # #
# # # # # import matplotlib.pyplot as plt
# # # # #
# # # # # # 模型和对应的指标
# # # # # models = [
# # # # #     "NNM+GCF(w/o CAE)",
# # # # #     "NNM+CAE(w/o GCF)",
# # # # #     "GCF+CAE(w/o NNM)",
# # # # #     "NNM+GCF+CAE"
# # # # # ]
# # # # # auc = [0.9736, 0.9691, 0.9778, 0.9864]
# # # # # aupr = [0.9693, 0.9649, 0.9745, 0.9843]
# # # # #
# # # # # x = range(len(models))
# # # # # width = 0.25  # 调整柱宽使图更美观
# # # # #
# # # # # # 找出最大值
# # # # # max_val = max(max(auc), max(aupr))
# # # # #
# # # # # # 创建图表
# # # # # fig, ax = plt.subplots(figsize=(10, 6))
# # # # # bars1 = ax.bar([i - width/2 for i in x], auc, width, label='AUC', color='orange')
# # # # # bars2 = ax.bar([i + width/2 for i in x], aupr, width, label='AUPR', color='orangered')
# # # # #
# # # # # # 添加数据标签，并加粗最大值
# # # # # for bar in bars1 + bars2:
# # # # #     yval = bar.get_height()
# # # # #     is_max = abs(yval - max_val) < 1e-4
# # # # #     fontweight = 'bold' if is_max else 'normal'
# # # # #     ax.text(
# # # # #         bar.get_x() + bar.get_width()/2,
# # # # #         yval + 0.0005,
# # # # #         f'{yval:.4f}',
# # # # #         ha='center',
# # # # #         va='bottom',
# # # # #         fontweight=fontweight
# # # # #     )
# # # # #
# # # # # # 设置Y轴范围以放大变化
# # # # # ax.set_ylim(0.96, 0.99)
# # # # #
# # # # # # 美化图表
# # # # # ax.set_xlabel('Models')
# # # # # ax.set_ylabel('Scores')
# # # # # ax.set_title('AUC and AUPR Comparison for HMDAD Dataset (Zoomed Y-axis)')
# # # # # ax.set_xticks(list(x))
# # # # # ax.set_xticklabels(models, rotation=15)
# # # # # ax.legend()
# # # # # plt.tight_layout()
# # # # #
# # # # # plt.show()
# # # #
# # # #
# # # # import matplotlib.pyplot as plt
# # # #
# # # # # 模型和对应的指标
# # # # models = [
# # # #     "NNM+GCF(w/o CAE)",
# # # #     "NNM+CAE(w/o GCF)",
# # # #     "GCF+CAE(w/o NNM)",
# # # #     "NNM+GCF+CAE"
# # # # ]
# # # # auc = [0.9736, 0.9691, 0.9778, 0.9864]
# # # # aupr = [0.9693, 0.9649, 0.9745, 0.9843]
# # # #
# # # # x = range(len(models))
# # # # width = 0.25  # 柱宽调整为更美观比例
# # # #
# # # # # 分别找出 AUC 和 AUPR 的最大值
# # # # max_auc = max(auc)
# # # # max_aupr = max(aupr)
# # # #
# # # # # 创建图表
# # # # fig, ax = plt.subplots(figsize=(10, 6))
# # # # bars1 = ax.bar([i - width/2 for i in x], auc, width, label='AUC', color='orange')
# # # # bars2 = ax.bar([i + width/2 for i in x], aupr, width, label='AUPR', color='orangered')
# # # #
# # # # # 添加数据标签，并加粗 AUC 和 AUPR 的最大值
# # # # for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
# # # #     # AUC 标签
# # # #     yval1 = bar1.get_height()
# # # #     fontweight1 = 'bold' if abs(yval1 - max_auc) < 1e-4 else 'normal'
# # # #     ax.text(
# # # #         bar1.get_x() + bar1.get_width()/2,
# # # #         yval1 + 0.0005,
# # # #         f'{yval1:.4f}',
# # # #         ha='center',
# # # #         va='bottom',
# # # #         fontweight=fontweight1
# # # #     )
# # # #
# # # #     # AUPR 标签
# # # #     yval2 = bar2.get_height()
# # # #     fontweight2 = 'bold' if abs(yval2 - max_aupr) < 1e-4 else 'normal'
# # # #     ax.text(
# # # #         bar2.get_x() + bar2.get_width()/2,
# # # #         yval2 + 0.0005,
# # # #         f'{yval2:.4f}',
# # # #         ha='center',
# # # #         va='bottom',
# # # #         fontweight=fontweight2
# # # #     )
# # # #
# # # # # 设置 Y 轴范围以放大可视差异
# # # # ax.set_ylim(0.96, 0.99)
# # # #
# # # # # 美化图表
# # # # ax.set_xlabel('Models')
# # # # # ax.set_ylabel('Scores')
# # # # # ax.set_title('AUC and AUPR Comparison for HMDAD Dataset ')
# # # # ax.set_xticks(list(x))
# # # # ax.set_xticklabels(models, rotation=15)
# # # # ax.legend()
# # # # plt.tight_layout()
# # # #
# # # # # 显示图表
# # # # plt.show()
# #
# #
# #
# # import matplotlib.pyplot as plt
# #
# # # 模型与指标
# # models = [
# #     "NNM+GCF(w/o CAE)",
# #     "NNM+CAE(w/o GCF)",
# #     "GCF+CAE(w/o NNM)",
# #     "NNM+GCF+CAE"
# # ]
# # auc = [0.9736, 0.9691, 0.9778, 0.9864]
# # aupr = [0.9693, 0.9649, 0.9745, 0.9843]
# #
# # x = range(len(models))
# # width = 0.15  # 更细的柱体
# #
# # # 图表设置
# # fig, ax = plt.subplots(figsize=(10, 6))
# #
# # # 偏移略微收紧距离
# # offset = width * 0.6
# # bars1 = ax.bar([i - offset for i in x], auc, width=width, label='AUC',
# #                color='royalblue', edgecolor='black', linewidth=1.3)
# # bars2 = ax.bar([i + offset for i in x], aupr, width=width, label='AUPR',
# #                color='skyblue', edgecolor='black', linewidth=1.3)
# #
# # # 添加数值标签（可选）
# # for bar in bars1 + bars2:
# #     height = bar.get_height()
# #     ax.text(bar.get_x() + bar.get_width() / 2, height + 0.0005,
# #             f'{height:.4f}', ha='center', va='bottom', fontsize=10)
# #
# # # 设置X轴
# # ax.set_xticks(list(x))
# # ax.set_xticklabels(models, rotation=12, fontsize=11)
# #
# # # Y轴缩放 + 去标签
# # ax.set_ylim(0.96, 0.99)
# # ax.set_ylabel('')  # 去掉Y轴标签
# # ax.tick_params(axis='y', labelsize=10)
# #
# # # 图例
# # ax.legend(fontsize=11)
# #
# # # 去除顶部和右侧边框，保持左和下边框
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# #
# # plt.tight_layout()
# #
# # # 保存为SVG矢量图（可选）
# # plt.savefig("grouped_model_barplot_final.svg", format='svg')
# #
# # plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据
# models = ["NNM+GCF(w/o CAE)", "NNM+CAE(w/o GCF)", "GCF+CAE(w/o NNM)", "NNM+GCF+CAE"]
# auc = [0.9736, 0.9691, 0.9778, 0.9864]
# aupr = [0.9693, 0.9649, 0.9745, 0.9843]
#
# # X轴位置
# x = np.arange(len(models))
# width = 0.18  # 柱子更细
# spacing = 0.03  # 两组之间的间距
#
# # 图形
# fig, ax = plt.subplots(figsize=(9, 5))
#
# # 绘制柱子
# bar_auc = ax.bar(x - width/2 - spacing/2, auc, width, label='AUC',
#                  color='#2b83ba', edgecolor='black', linewidth=1.3)
# bar_aupr = ax.bar(x + width/2 + spacing/2, aupr, width, label='AUPR',
#                   color='#abdda4', edgecolor='black', linewidth=1.3)
#
# # X轴设置
# ax.set_xticks(x)
# ax.set_xticklabels(models, rotation=10, fontsize=10)
#
# # 去除Y轴标签 & 设置范围
# ax.set_ylabel('')
# ax.set_ylim(0.96, 0.99)
#
# # 标题
# ax.set_title("Comparison of Models on AUC and AUPR", fontsize=12)
#
# # 图例（加粗字体）
# legend = ax.legend(fontsize=10)
# for text in legend.get_texts():
#     text.set_fontweight('bold')
#
# # 去除顶部和右侧边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
#
# plt.tight_layout()
# plt.savefig("final_model_comparison_boldlegend.svg", format="svg")
# plt.show()



import matplotlib.pyplot as plt

# 模型和对应的指标
models = [
    "NNM+GCF(w/o CAE)",
    "NNM+CAE(w/o GCF)",
    "GCF+CAE(w/o NNM)",
    "NNM+GCF+CAE"
]
auc = [0.9736, 0.9691, 0.9778, 0.9864]
aupr = [0.9693, 0.9649, 0.9745, 0.9843]

x = range(len(models))
width = 0.25  # 柱宽调整为更美观比例

# 分别找出 AUC 和 AUPR 的最大值
max_auc = max(auc)
max_aupr = max(aupr)

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar([i - width/2 for i in x], auc, width, label='AUC', color='orange')
bars2 = ax.bar([i + width/2 for i in x], aupr, width, label='AUPR', color='orangered')

# 添加数据标签，并加粗 AUC 和 AUPR 的最大值
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    # AUC 标签
    yval1 = bar1.get_height()
    fontweight1 = 'bold' if abs(yval1 - max_auc) < 1e-4 else 'normal'
    ax.text(
        bar1.get_x() + bar1.get_width()/2,
        yval1 + 0.0005,
        f'{yval1:.4f}',
        ha='center',
        va='bottom',
        fontweight=fontweight1
    )

    # AUPR 标签
    yval2 = bar2.get_height()
    fontweight2 = 'bold' if abs(yval2 - max_aupr) < 1e-4 else 'normal'
    ax.text(
        bar2.get_x() + bar2.get_width()/2,
        yval2 + 0.0005,
        f'{yval2:.4f}',
        ha='center',
        va='bottom',
        fontweight=fontweight2
    )

# 设置 Y 轴范围以放大可视差异
ax.set_ylim(0.96, 0.99)

# 美化图表
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('AUC and AUPR Comparison for HMDAD Dataset (Zoomed Y-axis)')
ax.set_xticks(list(x))
ax.set_xticklabels(models, rotation=15)
ax.legend()
plt.tight_layout()

# 显示图表
plt.show()
