from pylab import *
from nn import *

data = mat(loadtxt('addnum.txt'))
train_x, train_y = data[:800, 0:2], data[:800, -1]
test_x, test_y = data[800:, 0:2], data[800:, -1]

random.seed(0)
net = NN(train_x, train_y, [2, 10, 1])
#net.activefn = 'sigmoid'  #learn rate 1.0
net.activefn = 'sigmoid'
net.outfn = 'linear'
net.epoch = 100
net.learning_rate = 1.0
net.batch_size = 100

net.train()
loss, y = net.test(test_x, test_y)
print loss
plot(abs(y - test_y))
show()
