#!/usr/bin/python

import numpy as np
import pandas as pd


a = ['one','two','three']  
b = [1,2,3]  
#english_column = pd.Series(a, name='english')  
#number_column = pd.Series(b, name='number')  
#predictions = pd.concat([english_column, number_column], axis=1)  
#another way to handle  
save = pd.DataFrame({'english':a,'number':b})  
save.to_csv('csv_in_out.csv',index=False,sep=',') 

f=open('csv_in_out.csv')
df=pd.read_csv(f)

data = np.array(df['number'])
print data
