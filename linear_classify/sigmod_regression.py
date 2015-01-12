from pylab import *

def sigmod(x):
    return 1/1+exp(-x) 

data = np.loadtxt('data.txt')
num = data.shape[0]
print num
x1, x2, t = data[:,0].T, data[:,1].T, data[:,2].T
for i in range(num):
    if t[i] == 0:
        plot(x1[i], x2[i], 'ro')
    else:
        plot(x1[i], x2[i], 'bo')

#x = hstack((x1, x2, ones((num, 1))))
#w = ones((3, 1))
#epock = 1000
#for i in range(epock):
#   y = sigmod(dot(x, w))
#   dw = dot((y - t).T, x)
#   w = w + dw

show()
