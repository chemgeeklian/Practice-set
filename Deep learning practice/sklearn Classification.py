# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:55:10 2019
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

@author: Xinran Lian
"""

from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
np.random.seed(19260817)

class parameter():
    n=1000
    n_train=n-20
    threshold=(0.5/np.pi) #threshold of r^2, radius of circle (see the figure)
    
a=np.zeros([parameter.n,4])
a[:,:2]= np.random.rand(parameter.n,2) #initialize random scatters

a[:,2] = (a[:,0]-0.5)**2+(a[:,1]-0.5)**2>parameter.threshold
a[:,3] = a[:,0]>a[:,1] #bilabel classification

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 10), random_state=1)
clf.fit(a[:parameter.n_train,:2], a[:parameter.n_train,2:]) #fit neural network

predict=clf.predict(a[parameter.n_train:,:2]) #predict result
print(np.array(clf.predict(a[parameter.n_train:,:2]))==a[parameter.n_train:,2:])
#validate the prediction, print if the predicted label matches real label.

print([coef.shape for coef in clf.coefs_]) 
#the weight matrices that constitute the model parameters

#print(clf.predict_proba(a[parameter.n_train:,:2])) 
#P(y|x), probablity of y in case of x. Degree of certainty.

fig = plt.figure()
ax = fig.add_subplot(111)
for i,j,m in [(0,0,'r'),(0,1,'y'),(1,0,'g'),(1,1,'b')]:
    ax.scatter(a[np.where((a[:,2]==i)&(a[:,3]==j)),0], 
                 a[np.where((a[:,2]==i)&(a[:,3]==j)),1],s=10, c=m)
ax.scatter(a[parameter.n_train:,0], a[parameter.n_train:,1],s=20, c='black')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show() #visualize clustering results.

del i,j,m