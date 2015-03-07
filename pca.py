# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:17:33 2013
@author: robin1001
pca in pylab
"""

from pylab import *

m = floor(rand(10, 3) * 10)
#average
aver = mean(m, 0)
print 'mean'
print aver

for i in range(3):
    m[:, i] -= aver[i]

conv = dot(m.T, m)
d, v = linalg.eig(conv)
print 'covariance matrix'
print conv   
print 'eigenvalue and eigenvetor'
for i in range(len(d)):
    print d[i], v[i] 
