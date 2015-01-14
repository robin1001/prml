from numpy import *
from nn import *
data = mat(loadtxt('data.txt'))
x, y = data[:, 0:2], data[:, -1]


net = NN(x, y, [2, 3, 1], sigmoid, linear)
net.start()
