from numpy import *
from numpy.random import *

# author: robin1001 date: 2015-01-15

def sigmoid(x):
    return 1 / (1 + exp(-x))
def tanh(x):
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x))

class NN(object):
    def __init__(self, x, y, layer):
        ''' x: input 
            y: output 
            layer: layer structrue, eg: [400 100 10]
            outfn: output active funciton, eg: sigmoid, softmax, linear
            activefn: active function, eg: sigmoid, 
        '''
        self.activefn = 'sigmoid' 
        self.outfn = 'linear'
        self.x = mat(x)
        self.y = mat(y)
        self.layer = layer
        self.n = len(layer) #num of layer
        self.epoch = 1000
        self.learning_rate = 0.5
        self.batch_size = self.x.shape[0] #default 1 batch
        self.setup()

    def setup(self):
        '''setup the neural network, init w'''
        self.w = [None]*(self.n - 1)
        self.a = [None]*self.n
        for i in range(self.n - 1):
            n1, n2 = self.layer[i], self.layer[i+1]
            self.w[i] = mat(rand(n2, n1 + 1) - 0.5) / (n1 + n2)# 1 bias
            #self.w[i] = mat(rand(n2, n1 + 1)) # 1 bias

    def fp(self, x, y):
        '''forward propogation'''
        m = x.shape[0]
        self.a[0] = hstack((x, ones((m, 1))))
        for i in range(1, self.n - 1):
            if self.activefn == 'sigmoid':
                self.a[i] = sigmoid(self.a[i-1] * self.w[i-1].T)
            else:# tanh
                self.a[i] = tanh(self.a[i-1] * self.w[i-1].T)
            #add one bias 
            self.a[i] = hstack((self.a[i], ones((m, 1))))
        a = self.a[-2] * self.w[-1].T
        #switch output function
        if self.outfn == 'linear':
            self.a[-1] = a
            loss = (y - self.a[-1]).T * (y - self.a[-1]) / m
        elif self.outfn == 'sigmoid':
            self.a[-1] = sigmoid(a)
            loss = -sum(multiply(y, self.a[-1]) + multiply(1-y, 1-self.a[-1])) / m
        elif self.outfn == 'softmax':
            self.a[-1] = exp(a) / sum(exp(a), 1)
            loss = -sum(multiply(y, log(self.a[-1]))) 
        return loss, self.a[-1]
        
    def ebp(self, x, y):
        '''error backpropogation
           bishop prml neural network fomulation(5.50 - 5.56)
           dw: difference of w
           d: delta 
           h: difference of acitve function, eg:d(s(a)) = s(a) * (1 - s(a))
           sigmoid use cross entropy loss, linear output square loss
        '''
        dw, d = [None]*(self.n-1), [None]*self.n
        #switch output funciton
        d[-1] = -(y - self.a[-1])
        m = y.shape[0]
        for i in range(self.n - 2, 0, -1):
            #multiply fucntion: numpy dot multiply fucntion, like .* in matlab
            if self.activefn == 'sigmoid': h = multiply(self.a[i], self.a[i])
            else: h = 1 - multiply(self.a[i], self.a[i])
            if i+1 == self.n - 1: #last layer
                d[i] = multiply(h, d[i+1] * self.w[i])
            else:
                d[i] = multiply(h, d[i+1][:,:-1] * self.w[i])
        
        for i in range(0, self.n-1):
            if i+1 == self.n - 1: 
                dw[i] = (d[i+1].T * self.a[i]) / m
            else:
                dw[i] = (d[i+1][:,:-1].T * self.a[i]) / m
            #print i, self.check_gradient(i, dw[i], x, y)

        #gradient descent
        for i in range(0, self.n - 1):
            self.w[i] = self.w[i] - self.learning_rate * dw[i] 
    
               
    def train(self):
        m, n = self.x.shape
        num_batch, left = m / self.batch_size, m % self.batch_size
        if left != 0: num_batch += 1
        for i in range(self.epoch):
            start, total_loss = 0, 0
            for j in range(num_batch):
                end = start + self.batch_size
                if end > m: end = m
                loss, p = self.fp(self.x[start:end], self.y[start:end])
                self.ebp(self.x[start:end], self.y[start:end])
                print "epoch:%d minibatch:%d loss:%f " % (i, j, loss)
                total_loss += loss
                start += self.batch_size
            print "epoch:%d total_loss:%f\n" % (i, total_loss)

    def check_gradient(self, k, dw, x, y):
        ''' check gradient of k th w 
            coursera standord ng machine learning chapter 9-5'''
        eps = 0.0001
        m, n = dw.shape
        d = mat(zeros(dw.shape))
        for i in range(m):
            for j in range(n):
                w = self.w[k][i,j]
                self.w[k][i,j] = w - eps 
                l1, p = self.fp(x, y)
                self.w[k][i,j] = w + eps 
                l2, p = self.fp(x, y) 
                d[i, j] = (l2 - l1) / (2 * eps)
                self.w[k][i,j] = w #keep self.w invariant  
        err = d - dw
        return sum(multiply(err, err)) 

    def test(self, x, y):
        loss, p = self.fp(mat(x), mat(y))
        return loss, p
