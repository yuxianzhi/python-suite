#!/usr/bin/python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f=open('data.csv')
df=pd.read_csv(f)     
data=np.array(df['price'])   
data=data[::-1]   
plt.figure()
plt.plot(data)
plt.show()
