from numpy import *
from nn import *

data = mat(loadtxt('data.txt'))
train_x, train_y = data[:800, 0:2], data[:800, -1]
test_x, test_y = data[800:, 0:2], data[800:, -1]


net = NN(train_x, train_y, [2, 3, 1], sigmoid, linear)
net.train()
net.test_regression(test_x, test_y)
