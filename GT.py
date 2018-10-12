# -*- coding: utf-8 -*-
"""
Proof of Gerschgorm's theorem

Xinran Lian 
Oct 12, 2018
"""

import numpy as np
from numpy import linalg as LA
np.random.seed(19260817)

class para:
    n=10

def create_complex_mat(n):
    A_r = np.random.rand(n,n)
    A_i = np.random.rand(n,n)*1j
    return(A_r + A_i)
    
within = lambda z,r:LA.norm(z)<r
A=create_complex_mat(para.n)
w, v = LA.eig(A)

whether = np.zeros(para.n) ==1
test=np.arange(3,3+para.n) #to test the eigenvalues, test = w
for i in range(para.n):
    for j in range(para.n):
        if within(abs(test[i]),sum(abs(A[j]))-abs(A[j][j]))==True:
            whether[i]=1
            break

print(whether)