# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 

k=[0,0,0]
a=[0,0,0]
b=[0,0,0]
k[0]=0.14307295
a[0]=0.18259039
b[0]=0.21640213
#假设过0.5天复习一次,过60天再复习一次(即期末复习),第66天考试
tt=[0,0.5,60]

def func(i,x):
    y=k[i]/(x+a[i])+ b[i]
    return y

for i in range(1,3):
    k[i]=k[i-1]/2
    b[i]=b[i-1]+(1-b[i-1])/2
    a[i]=k[i]/(1-b[i])-tt[i]
t0=np.arange(0,0.5,0.1)
t1=np.arange(0.5,60,0.1)
t2=np.arange(60,66,0.1)

beautifulx=np.array([60,60])
beautifuly=np.array([func(1,tt[2]),func(2,tt[2])])
beautifulx0=np.array([0.5,0.5])
beautifuly0=np.array([func(0,tt[1]),func(1,tt[1])])
'''
其实这个可以用photoshop...
但是明明是用photoshop画上的两条直线却说这个图表是用程序写的,
会不会给人一种欺骗老师的感觉?
另外这个画图程序有点复杂,希望老师指正
'''
fig = plt.figure()
plt.plot(t0,func(0,t0),linewidth=2,color='green')
plt.plot(t1,func(1,t1),linewidth=2,color='green')
plt.plot(t2,func(2,t2),linewidth=2,color='green')
plt.plot(beautifulx,beautifuly,linewidth=2,color='green')
plt.plot(beautifulx0,beautifuly0,linewidth=2,color='green')
plt.title('my review curve')
plt.xlabel('t/d')
plt.ylabel('ratio')

print(func(2,66))
print(func(0,10))
#输出0.8098858536030598;0.23045287195467073
