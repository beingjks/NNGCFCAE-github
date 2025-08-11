import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [0.9770,0.8693,0.9088]

# 创建图形和坐标系
fig, ax = plt.subplots()

# 设置横坐标的值和显示的值
ax.plot(x, y, marker='o', linestyle='')
ax.plot(x, y)
ax.set_xticks(x)  # 设置横坐标显示的位置
ax.set_xticklabels(['1', '2', '3'])
ax.set_xlabel('GCAN_l')
ax.set_ylabel('AUC value')

# 设置横坐标显示的值

# 显示图形
plt.show()
