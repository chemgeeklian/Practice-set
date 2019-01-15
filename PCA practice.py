# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
np.random.seed(19260817)
#fix the random seed to keep output diagrams the same every time.

class parameter():
    nx=8
    ny=10
    distort=8
    dim=3

#Pertube scatters distributed on a plan (in 3D space)
def pertub(x):
    p=(np.random.random_sample(np.shape(x))-0.5)*parameter.distort
    p=p.reshape(np.shape(x))
    return(x+p)

def PCA(dataset,high_d,lower_d): #lower dimension must be lower than
    space=dataset.reshape(high_d,-1).transpose()
    space_centr=space-np.mean(space,axis=0) #centralize dataset
    m=np.size(space,axis=0)
    cov=np.dot(space_centr.transpose(),space_centr)/m #cov(X,X)
    
    eign,vector=np.linalg.eig(cov) 
    eignandvector=np.append(eign,vector).reshape(-1,high_d)
    eignandvector_sort=eignandvector.T[eignandvector.T[:,0].argsort(kind='mergesort')].T
    eignandvector_sort=np.flip(eignandvector_sort,axis=1)
    #Sort the eigen values (from large to small) 
    #and corresponding eigen vectors. merge sort is more stable.
    
    new_base = eignandvector_sort[1:,:lower_d] 
    #choose the eigen vectors corresponding to largest eigen values as new basis
    dataset_new=np.dot(new_base.transpose(),space_centr.transpose())
    dataset_new=dataset_new.reshape(lower_d,-1)
    return (new_base,dataset_new)

################m generate a dataset ########################################
x, y = np.arange(parameter.distort + parameter.nx+parameter.distort) ,\
                 np.arange(parameter.ny)
x, y = np.meshgrid(x,y)
dataset = np.array([x,y,x+y])

for i in range(np.size(dataset,axis=0)):
    dataset[i]=pertub(dataset[i])
#############################################################################

new_base,test=PCA(np.array([[2,-1,1],[-1,5,0],[1,0,3]]),3,3)#(dataset,np.size(dataset,axis=0),2)

'''
#Plot the 3D figure begore PCA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Plot the two new bases determined by PCA
ax.scatter(dataset[0], dataset[1], dataset[2], c='r', marker='o')
new_base*=parameter.distort
ax.plot([0,new_base[0,1]],[0,new_base[1,1]],[0,new_base[2,1]],  c='b')
ax.plot([0,new_base[0,0]],[0,new_base[1,0]],[0,new_base[2,0]],  c='b')
    
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

#plot the 2D figure after PCA
plot1=plt.plot(test[0],test[1],'o',c='r')'''