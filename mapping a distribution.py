# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:54:09 2019

@author: Administrator

# mapping the normal distribution a to X = g(a) to form a ring
"""

import numpy as np
import matplotlib.pyplot as plt

a=np.random.normal(loc=0.0, scale=2.0,size = [2,200])-0.5
g_a=a/10 + a/np.linalg.norm(a,axis=0)

fig, (axs1,axs2) = plt.subplots(1,2)
axs1.plot(a[0,:],a[1,:],'o')
axs1.set_aspect('equal', 'box')
axs1.set_title('a')

axs2.plot(g_a[0,:],g_a[1,:],'o')
axs2.set_aspect('equal', 'box')
axs2.set_title('g(a)')
plt.show()