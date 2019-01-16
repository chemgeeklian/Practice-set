# -*- coding: utf-8 -*-
"""
k-mean cluster

Reference: https://en.wikipedia.org/wiki/K-means_clustering
Not perfect version. The results could be unsatisfied if the initial centers are
not properly picked. But currently I don't know how to improve it...

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
np.random.seed(19260817)

class parameter():
    n=np.array([50,30,40,45]) #np.sum(parameter.n)=
    dim=2
    guess_nk=4
    max_iter=50

#generate 3D clusters
a=np.array([-1,1])+np.random.rand(parameter.n[0],parameter.dim)
b=np.array([-0.5,-1]) +np.random.rand(parameter.n[1],parameter.dim)
c=np.array([0.5,0.5]) +np.random.rand(parameter.n[2],parameter.dim)
d=np.array([1,-1])+np.random.rand(parameter.n[3],parameter.dim)

#convert the clusters into an unlabeled set.
dataset=np.concatenate([a,b,c,d],axis=0)
dataset=np.c_[dataset,np.zeros(np.sum(parameter.n))]
df = pd.DataFrame(dataset,columns=['x', 'y', 'label'])
df = shuffle(df)
df.index = range(len(df))

#randomly pick up four scatter points to initialize kmean.
kmean=df.loc[np.random.randint(0,np.sum(parameter.n),parameter.guess_nk),['x','y']]
#kmean=pd.DataFrame([[-0.5,1.5,-0.5],[0.5,-0.5,2],[1.5,0.5,1.5],[1.5,-0.5,0.5]],columns=['x', 'y', 'z'])
kmean.index = range(len(kmean))
kmean_old=kmean.copy()

fig = plt.figure()
ax = fig.add_subplot(111)
for i, m in [(a,'r'),(b,'y'),(c,'g'),(d,'b')]:
    ax.scatter(i[:,0], i[:,1], c=m)
ax.scatter(kmean.loc[:,'x'], kmean.loc[:,'y'], c='black')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show() #visualize randomly generated four clusters

for i in range(parameter.max_iter):
    for j in range(np.sum(parameter.n)):
        point=np.array(df.loc[j,['x','y']])
        norms=np.linalg.norm(point-kmean,axis=1)
        df.loc[j,'label']=np.argmin(norms)
    kmean_old=kmean
    for k in range(parameter.guess_nk):
        kmean.loc[k]=df.loc[np.array(np.where(df['label']==k)).reshape(-1).
                 tolist(),['x','y']].mean()
    #if max(np.linalg.norm(kmean-kmean_old,axis=1))<1e-15:
    #   break

fig = plt.figure()
ax = fig.add_subplot(111)
for i, m in [(0,'r'),(1,'y'),(2,'g'),(3,'b')]:
    ax.scatter(df.loc[np.where(df['label']==i)]['x'], 
                      df.loc[np.where(df['label']==i)]['y'], c=m)
ax.scatter(kmean.loc[:,'x'], kmean.loc[:,'y'], c='black')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show() #visualize clustering results.