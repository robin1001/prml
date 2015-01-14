from numpy import *
from numpy.random import *

def sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

def tanh(x):
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def linear(x):
	return x

class NN(object):
	def __init__(self, x, y, layer, activefn=sigmoid, outfn=sigmoid):
		'''	x: input 
			y: output 
			layer: layer structrue, eg: [400 100 10]
			outfn: output active funciton, eg: sigmoid, softmax, linear
			activefn: active function, eg: sigmoid, 
		'''
		self.outfn = outfn
		self.activefn = activefn
		self.x = mat(x)
		self.y = mat(y)
		self.layer = layer
		self.n = len(layer) #num of layer
		self.epoch = 100
		self.alpha = 0.01
		self.setup()

	def setup(self):
		'''setup the neural network, init w'''
		self.w = [None]*(self.n - 1)
		self.a = [None]*self.n
		for i in range(self.n - 1):	
			self.w[i] = mat(rand(self.layer[i+1], self.layer[i] + 1)) # 1 bias
	def fp(self):
		'''forward propogation'''
		m = self.x.shape[0]
		self.a[0] = hstack((self.x, ones((m, 1))))
		for i in range(1, self.n - 1):
			self.a[i] = self.activefn(self.a[i-1] * self.w[i-1].T)
			#add one bias 
			self.a[i] = hstack((self.a[i], ones((m, 1))))
		#switch output function
		print self.a[-2].shape, self.w[-1].shape
		self.a[-1] = self.outfn(self.a[-2] * self.w[-1].T)
		e = self.y - self.a[-1]
		self.L = e.T * e
		
	def ebp(self):
		'''error backpropogation
		   bishop prml neural network fomulation(5.50 - 5.56)
		   dw: difference of w
		   d: delta 
		   h: difference of acitve function, eg:d(s(a)) = s(a) * (1 - s(a))
		'''
		dw, d = [None]*(self.n-1), [None]*self.n
		#switch output funciton
		d[-1] = self.y - self.a[-1]
		for i in range(self.n - 2, 1, -1):
		   	#multiply fucntion: numpy dot multiply fucntion, like .* in matlab
			h = multiply(self.a[i], 1 - self.a[i]) 
			if i+1 == self.n - 1: #last layer
				d[i] = multipy(h, d[i+1] * self.w[i])
			else:
				d[i] = multipy(h, d[i+1][:,:-1] * self.w[i])
		
		for i in range(0, self.n-1):
			if i+1 == self.n - 1: 
				dw[i] = d[i+1].T * self.a[i]
			else:
				dw[i] = d[i+1][:,:-1].T * self.a[i]
		#gradient descent
		for i in range(0, self.n - 1):
			self.w = self.w - self.alpha * dw[i]

	def start(self):
		self.fp()
		self.ebp();
