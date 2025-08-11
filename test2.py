
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# 从文件加载fpr, tpr, precision, recall
fpr = np.loadtxt("fpr_tpr GCAT=0.05SAE=0.01/fpr0.txt")  # 替换为你的实际路径
tpr = np.loadtxt("fpr_tpr GCAT=0.05SAE=0.01/tpr0.txt")  # 替换为你的实际路径

#计算AUC
roc_auc = auc(fpr, tpr)


# 绘制ROC曲线
plt.figure(figsize=(10, 5))

# ROC曲线
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')


# 显示图形
plt.tight_layout()
plt.show()


# 原来的auc 在0.93 左右
