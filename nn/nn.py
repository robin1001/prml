from numpy import *
from numpy.random import *

def sigmoid(msg, x):
    if msg == 'fp': return 1.0 / (1.0 + exp(-x))
    elif msg == 'bp': return multiply(x, 1 - x)
    elif msg == 'loss': pass        
        
def tanh(msg, x):
    if msg == 'fp': return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    elif msg == 'bp': return 1 - multiply(x, x)

def linear(msg, x):
    if msg == 'fp': return x
    elif msg == 'loss': pass
    elif msg == 'bp': return -(x[0] - x[1])

def softmax(msg, x):
    if msg == 'fp':  pass
    elif msg == 'bp': pass
    elif msg == 'loss': pass

class NN(object):
    def __init__(self, x, y, layer, activefn=sigmoid, outfn=sigmoid):
        ''' x: input 
            y: output 
            layer: layer structrue, eg: [400 100 10]
            outfn: output active funciton, eg: sigmoid, softmax, linear
            activefn: active function, eg: sigmoid, 
        '''
        self.activefn = activefn 
        self.outfn = outfn
        self.x = mat(x)
        self.y = mat(y)
        self.layer = layer
        self.n = len(layer) #num of layer
        self.epoch = 1000
        self.learning_rate = 0.5
        self.setup()

    def setup(self):
        '''setup the neural network, init w'''
        self.w = [None]*(self.n - 1)
        self.a = [None]*self.n
        for i in range(self.n - 1): 
            self.w[i] = mat(rand(self.layer[i+1], self.layer[i] + 1)) # 1 bias

    def fp(self, x, y):
        '''forward propogation'''
        m = x.shape[0]
        self.a[0] = hstack((x, ones((m, 1))))
        for i in range(1, self.n - 1):
            self.a[i] = self.activefn('fp', self.a[i-1] * self.w[i-1].T)
            #add one bias 
            self.a[i] = hstack((self.a[i], ones((m, 1))))
        #switch output function
        self.a[-1] = self.outfn('fp', self.a[-2] * self.w[-1].T)
        e = y - self.a[-1]
        loss = (e.T * e) / m
        return loss
        
    def ebp(self):
        '''error backpropogation
           bishop prml neural network fomulation(5.50 - 5.56)
           dw: difference of w
           d: delta 
           h: difference of acitve function, eg:d(s(a)) = s(a) * (1 - s(a))
        '''
        dw, d = [None]*(self.n-1), [None]*self.n
        #switch output funciton
        d[-1] = self.outfn('bp',(self.y, self.a[-1]))
        m = self.y.shape[0]
        for i in range(self.n - 2, 0, -1):
            #multiply fucntion: numpy dot multiply fucntion, like .* in matlab
            h = self.activefn('bp', self.a[i])
            if i+1 == self.n - 1: #last layer
                d[i] = multiply(h, d[i+1] * self.w[i])
            else:
                d[i] = multiply(h, d[i+1][:,:-1] * self.w[i])
        
        for i in range(0, self.n-1):
            if i+1 == self.n - 1: 
                dw[i] = (d[i+1].T * self.a[i]) / m
            else:
                dw[i] = (d[i+1][:,:-1].T * self.a[i]) / m
            #print i, self.check_gradient(i, dw[i])

        #gradient descent
        for i in range(0, self.n - 1):
            self.w[i] = self.w[i] - self.learning_rate * dw[i] 
    
               
    def train(self):
        for i in range(self.epoch):
            loss = self.fp(self.x, self.y)
            self.ebp()
            print "loss", loss

    def check_gradient(self, k, dw):
        ''' check gradient of k th w 
            coursera standord ng machine learning chapter 9-5'''
        eps = 0.0001
        m, n = dw.shape
        d = mat(zeros(dw.shape))
        for i in range(m):
            for j in range(n):
                w = self.w[k][i,j]
                self.w[k][i,j] = w - eps 
                l1 = self.fp(self.x, self.y)
                self.w[k][i,j] = w + eps 
                l2 = self.fp(self.x, self.y) 
                d[i, j] = (l2 - l1) / (2 * eps)
                self.w[k][i,j] = w #keep self.w invariant  
        err = d - dw
        return sum(multiply(err, err)) 

    def test_regression(self, x, y):
        self.fp(mat(x), mat(y))
        #com = hstack((y, self.a[-1]))
        return self.a[-1]

    def test_classify(self, x, y):
        self.fp(mat(x), mat(y))


