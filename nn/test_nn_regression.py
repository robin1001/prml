from pylab import *
from nn import *

data = mat(loadtxt('data_regression_1000.txt'))
train_x, train_y = data[:800, 0:2], data[:800, -1]
test_x, test_y = data[800:, 0:2], data[800:, -1]


net = NN(train_x, train_y, [2,  10, 1], sigmoid, linear)
net.epoch = 1000
net.learning_rate = 0.5

net.train()
y = net.test_regression(test_x, test_y)
plot(y - test_y)
show()
