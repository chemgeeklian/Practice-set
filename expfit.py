#和power fit差不多,拟合公式和参数改了一下而以
# -*- coding: utf-8 -*-

from scipy import exp as exp
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b,k,c):
    y = k*exp(-a*(x-c)) + b
    return y

def polyfit(x, y, degree):
    results = {}
    #coeffs = numpy.polyfit(x, y, degree)
    popt, pcov = curve_fit(func, x, y)
    results['polynomial'] = popt

    # r-squared
    yhat = func(x ,popt[0] ,popt[1],popt[2],popt[3] )                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

x=[ 0 ,1, 2 ,6, 30]
y=[ 1 ,0.337 ,0.278,0.254,0.211]
z1 = polyfit(x, y,2)
print(z1)

q=np.arange(0,30,0.1)
r= 0.30499393*exp(-0.75843018*(q-0.45421855))+0.24115065
    
import matplotlib.pyplot as plt 

fig = plt.figure()
plt.plot(x,y,color='red',linewidth=2)
plt.plot(q,r,linewidth=2)
plt.xlabel('t/d')
plt.ylabel('ratio')
plt.title('Curve fit')
plt.show()
