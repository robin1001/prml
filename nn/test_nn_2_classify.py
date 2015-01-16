from pylab import *
from nn import *
import scipy.io 

data = mat(loadtxt('data.txt'))
dim_x = 2
train_num = 1000
output = 2
train_x, train_y = data[:train_num, :dim_x], data[:train_num, dim_x:]
test_x, test_y = data[train_num:, :dim_x], data[train_num:, dim_x:]

if output != 1: 
    train_y = hstack((train_y, 1-train_y))
    test_y = hstack((test_y, 1-test_y))
net = NN(train_x, train_y, [2, 10, output])
net.activefn = 'sigmoid'  #learn rate 1.0
#net.activefn = 'tanh'
net.outfn = 'softmax'
#net.outfn = 'sigmoid'
net.epoch = 100
net.batch_size = 10
net.learning_rate = 2.5

net.train()
loss, y = net.test(test_x, test_y)
if output == 1:
    pred_y = y > 0.5
    print mean(pred_y == test_y)
    pos = find(y > 0.5)
    neg = find(y <= 0.5)
else:
    in1 = argmax(test_y, 1)    
    in2 = argmax(y, 1)
    print mean(in1 == in2)
    pos = find(y[:,0] > y[:,1])
    neg = find(y[:,0] <= y[:,1])
a = arange(0, 1, 0.1)
plot(a, a)
plot(test_x[pos, 0], test_x[pos, 1], 'ro')
plot(test_x[neg, 0], test_x[neg, 1], 'go')
show()
