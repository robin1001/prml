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
    def __init__(self, in_dim, out_dim, opts=Option()):
        self.w, self.dw = randn(out_dim, in_dim), zeros((out_dim, in_dim))
        self.b = rand(1, out_dim), zeros((1, out_dim))
        self.opts = opts
    def foward_propagate(self, inn):
        return sigmoid(inn * self.w.T + self.b)
    def back_propagate(self, inn, out, out_diff):
        return multiply(multiply(out, 1 - out), out_diff) * self.w
    def update(self, diff):
        

class SoftmaxLayer:
    pass

class Net:
