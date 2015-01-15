from pylab import *
from nn import *
import scipy.io 


data = scipy.io.loadmat('mnist_uint8.mat')
train_num,  test_num = 1000, 200
train_x, train_y = data['train_x'][:train_num], data['train_y'][:train_num]
test_x, test_y = data['test_x'][:test_num], data['test_y'][:test_num]

img = train_x[1,:].copy().reshape(28,28)
imshow(img, cmap=plt.cm.gray)

net = NN(train_x, train_y, [784, 100, 10])
net.activefn = 'sigmoid'  #learn rate 1.0
#net.activefn = 'tanh'
#net.outfn = 'softmax'
net.outfn = 'sigmoid'
net.epoch = 1000
net.learning_rate = 1.0
#
net.train()
loss, y = net.test(test_x, test_y)
in1 = argmax(test_y, 1)
in2 = argmax(y, 1)
print mean(in1 == in2)
#plot(abs(y - test_y))
#show()