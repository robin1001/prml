from numpy import *
from numpy.random import *

def sigmoid(x):
    return 1 / (1 + exp(-x))
def tanh(x):
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x))

class Option:
    def __init__(self, learn_rate=0.05, momentum=0):
        self.learn_rate = learn_rate
        self.momentum = momentum

class SigmoidLayer:
    '''pdw, pdb: privious dw db'''
    def __init__(self, in_dim, out_dim, opts=Option()):
        self.w, self.pdw = randn(out_dim, in_dim), zeros((out_dim, in_dim))
        self.b, self.pdb = rand(1, out_dim), zeros((1, out_dim))
        self.opts = opts
    def foward_propagate(self, inn):
        return sigmoid(inn * self.w.T + self.b)
    def back_propagate(self, inn, out, out_diff):
        d = multiply(multiply(out, 1 - out), out_diff)
        in_diff = d * self.w
        dw, db= d.T * inn, d.sum(1)
        learn_rate, momentum = self.opts.learn_rate, self.opts.momentum 
        self.w = self.w - learn_rate * dw + momentum * self.pdw
        self.b = self.b - learn_rate * db + momentum * self.pdb
        self.pdw, self.pdb = db, dw
        return in_diff
        
class SoftmaxLayer:
    pass

class Net:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.forward_buf = [None] * self.num_layers
        self.back_buf = [None] * self.num_layers
    def foward_propagate(self, inn):
    def back_propagate(self, inn, t):
    
