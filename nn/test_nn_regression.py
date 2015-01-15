from pylab import *
from nn import *

data = mat(loadtxt('addnum.txt'))
train_x, train_y = data[:800, 0:2], data[:800, -1]
test_x, test_y = data[800:, 0:2], data[800:, -1]


net = NN(train_x, train_y, [2, 3, 1])
net.activefn = 'sigmoid'  #learn rate 1.0
#net.activefn = 'tanh'
net.outfn = 'linear'
net.epoch = 1000
net.learning_rate = 1.0
net.batch_size = 10

net.train()
loss, y = net.test(test_x, test_y)
#plot(abs(y - test_y))
print abs(y - test_y)
#show()
