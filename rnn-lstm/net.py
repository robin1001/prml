class Net:
    def __init__(self, layers):
        self.layers = layers
        self.forward_buf = [None] * len(self.layers)
        self.back_buf = [None] * len(self.layers)
    def foward_propagate(self, inn):
    def back_propagate(self, inn, t):
 
