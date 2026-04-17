import numpy as np

class Activation:
    def __call__(self, x):
        return self.forward(x)

class ReLU(Activation):
    def forward(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)

class Sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)

class Tanh(Activation):
    def forward(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x)**2

class Softmax(Activation):
    def forward(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    def derivative(self, x):
        # Softmax derivative is complex for general case, 
        # usually handled with Cross Entropy.
        return None
