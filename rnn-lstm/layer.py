from numpy import *
from numpy.random import *

def sigmoid(x):
    return 1 / (1 + exp(-x))
def tanh(x):
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x))

class FullLayer:
    '''pdw, pdb: privious dw db'''
    def __init__(self, in_dim, out_dim, activefn, learn_rate, momentum=0) 
        self.w, self.pdw = randn(out_dim, in_dim), zeros((out_dim, in_dim))
        self.b, self.pdb = rand(1, out_dim), zeros((1, out_dim))
        self.activefn = activefn
        self.learn_rate, self.momentum = learn_rate, momentum

    def foward_propagate(self, inn): pass
        act = inn * self.w.T + self.b
        if self.activefn == 'sigmoid':     
            return sigmoid(act)
        elif self.activefn == 'tanh':      
            return tanh(act)
        elif self.activefn == 'linear':    
            return act
        elif self.activefn == 'softmax':   
            a = exp(act)
            return a / a.sum(1)
        
    def back_propagate(self, inn, out, out_diff): 
        if self.activefn == 'sigmoid':     
            act_diff = multiply(multiply(out, 1 - out), out_diff)
        elif self.activefn == 'tanh':      
            act_diff = multiply((1-multiply(self.a[i], self.a[i])) , out_diff)
        elif self.activefn == 'linear' or self.activefn == 'softmax': 
            act_diff = out_diff
        in_diff = act_diff * self.w
        dw, db= act_diff.T * inn, act_diff.sum(1)
        learn_rate, momentum = self.opts.learn_rate, self.opts.momentum 
        self.w = self.w - self.learn_rate * dw + self.momentum * self.pdw
        self.b = self.b - self.learn_rate * db + self.momentum * self.pdb
        self.pdw, self.pdb = db, dw
        return in_diff

def RecurrentLayer:
    def type(self):
        return 'recurrent_layer'

def LstmLayer:
    def type(self):
        return 'lstm_layer'
