from pylab import *

def sigmod(x):
    return 1.0 / (1 + exp(-x))

data = mat(np.loadtxt('data.txt'))
num = data.shape[0]

x = hstack((data[:, 0:2], ones((num, 1))))
t = data[:, 2]
w = ones((3, 1))
epock = 1000
alpha = 0.001
for i in range(epock):
   y = sigmod(x * w)
   dw = x.T * (t - y) 
   w = w + alpha * dw

class0 = data[find(data[:, 2] == 0)]
class1 = data[find(data[:, 2] == 1)]
plot(class0[:, 0], class0[:, 1], 'ro')
plot(class1[:, 0], class1[:, 1], 'go')

y = -(w[2,0] + w[0,0] * x[:, 0]) / w[1,0]
plot(x[:, 0], y)
show()

