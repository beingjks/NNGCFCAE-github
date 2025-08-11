import numpy as np
import pandas as pd

microbe=pd.read_excel("Data/Microbe_Number.xlsx")
microbe=microbe.values
disease=pd.read_excel("Data/Disease_Number.xlsx")
disease=disease.values
score=np.loadtxt("score/score0.txt")
disease1=list(score[3,:])     #Asthma
disease2=list(score[28,:])     #obesity
disease3=list(score[36,:])     #T2D
print(max(score[3,:]))
print(min(score[3,:]))
md1=dict()
md2=dict()
md3=dict()


for i in range(len(disease1)):
    md1[i]=disease1[i]
    md2[i]=disease2[i]
    md3[i]=disease3[i]



dm1=sorted(md1.items(),key=lambda x : x[1],reverse=True)
dm2=sorted(md2.items(),key=lambda x : x[1],reverse=True)
dm3=sorted(md3.items(),key=lambda x : x[1],reverse=True)

d1=[]
d2=[]
d3=[]


for i in range(50):

    d1.append(microbe[dm1[i][0]][1])
    d2.append(microbe[dm2[i][0]][1])
    d3.append(microbe[dm3[i][0]][1])


np.savetxt("prediction/microbe1_drug.txt",d1,fmt="%s")
np.savetxt("prediction/microbe2_drug.txt",d2,fmt="%s")
np.savetxt("prediction/microbe3_drug.txt",d3,fmt="%s")
