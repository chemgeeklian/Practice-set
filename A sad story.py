# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:24:09 2019

@author: Administrator
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
np.random.seed(19260817)

n=1000
threshold=0.3
size_plt=6

a=np.zeros([n,3])
a[:,:2]= np.random.rand(n,2)*3
keep=np.where((a[:,0]+a[:,1]>2)&(a[:,0]+a[:,1]<3)&(np.abs(a[:,0]-a[:,1])<1))
a[keep,2]=1
a=a[np.where(a[:,2]==1)]

rich=np.where(a[:,1]-a[:,0]>threshold)
poor=np.where(a[:,1]-a[:,0]<-threshold)
a[rich,2]=0
a[poor,2]=2

with plt.xkcd():
    fig = plt.figure()
    fig.set_size_inches(size_plt, size_plt)
    ax = fig.add_subplot(111)
    ax.scatter(a[:,0], a[:,1],s=20, c='black')
    ax.set_xlabel('Degree of Hard-working')
    ax.set_ylabel('Income')
plt.show()

with plt.xkcd():
    fig2 = plt.figure()
    fig2.set_size_inches(size_plt, size_plt)
    ax = fig2.add_subplot(111)
    for i,m in [(0,'r'),(1,'y'),(2,'g')]:
        ax.scatter(a[np.where(a[:,2]==i),0], 
                 a[np.where(a[:,2]==i),1],s=20, c=m)
        ax.set_xlabel('Degree of Hard-working')
        ax.set_ylabel('Income')
        ax.legend(['landlord', 'computer science', 'chemistry'])
plt.show()