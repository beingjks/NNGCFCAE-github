import numpy as np
# print(np.__version__)
# a= 1
# print("a:",a)
num =str(1)
precision =[1,3]
recall =[2,4]
np.savetxt("Dis_roc_pr/precision" + num + ".txt", precision)
np.savetxt("Dis_roc_pr/recall" + num + ".txt", recall)