from pylab import *

def sigmod(x):
    return 1.0 / (1 + exp(-x))

data = mat(np.loadtxt('data.txt'))
num = data.shape[0]
print num
x1, x2, t = data[:,0], data[:,1], data[:,2]
for i in range(num):
    if t[i] == 0:
        plot(x1[i], x2[i], 'ro')
    else:
        plot(x1[i], x2[i], 'bo')

x = hstack((x1, x2, ones((num, 1))))
w = ones((3, 1))
epock = 1000
alpha = 0.001
for i in range(epock):
   y = sigmod(x * w)
   dw = x.T * (y - t) 
   w = w + multiply(alpha, dw)

print w
y = mat(zeros((num, 1)))
for i in range(num):
	y[i] = -(w[0] * x1[i] + w[2]) / w[1]
yy = hstack((x2, y))
print yy
show()
