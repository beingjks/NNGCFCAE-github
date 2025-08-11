import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
plt.switch_backend('Tkagg')

#取最好的auc值画ROC曲线
a=np.loadtxt("AUC--------CSAE/auc.txt")
b=dict()
for i in range(len(a)):
    b[i]=a[i]
num=[k for k,v in b.items() if v==max(a)]
maxindex=num[0]

fpr=np.loadtxt("fpr_tpr/fpr"+str(maxindex)+".txt")
tpr=np.loadtxt("fpr_tpr/tpr"+str(maxindex)+".txt")
#fpr=np.loadtxt("fpr_tpr/fpr2"+".txt")
#tpr=np.loadtxt("fpr_tpr/tpr2"+".txt")
auc_val=auc(fpr,tpr)
print(auc_val)
plt.figure()
plt.plot(fpr,tpr)
plt.show()