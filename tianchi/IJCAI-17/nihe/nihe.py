#!/usr/bin/python

import numpy as np
import pandas as pd
import sys
import warnings
warnings.simplefilter('ignore', np.RankWarning)

f=open('data.csv')
df=pd.read_csv(f)
data = np.array(df['renshu'])
#print data
data_mean=np.mean(data)
data_std=np.std(data)
normalized_data = (data - data_mean) / data_std


data_num = len(normalized_data)
predict_num=14
jieshu=50
factor=1

x1=[]
for j in range(data_num):
	x1.append(j*factor)
x2=[] 
for j in range(predict_num):
        x2.append((j+data_num)*factor)
error=[0]*jieshu


for j in range(jieshu):
        Y_xishu=np.polyfit(x1, normalized_data, j)
	for i in range(data_num):
		y=np.polyval(Y_xishu,x1[i])
		ori_y=y*data_std + data_mean
		ori_y=max(0, ori_y)
        	error[j]=error[j]+(abs(ori_y-data[i])/(ori_y+data[i]))

jie_opt=1
for j in range(jieshu):
        if error[j]<error[jie_opt] :
            	jie_opt = j

Y_xishu=np.polyfit(x1, normalized_data, jie_opt);
Y2=[]
for i in range(predict_num):
	Y2.append(np.polyval(Y_xishu,x2[i]))

def array_out(array):
        dianjia=sys.argv[1]

        f = file('predict.csv','aw')
        f.write(dianjia)
        for i in array:
                f.write(",")
                f.write(str(int(max(0,round(i*np.std(data)+np.mean(data))))))
        f.write('\n')
        f.close()

array_out(Y2)
