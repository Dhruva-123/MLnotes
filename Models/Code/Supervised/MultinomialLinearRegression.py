import numpy as np
class MLG:
    def __init__(self , n_iter, eta):
        self.eta = eta
        self.n_iter = n_iter        
    
    def fit(self, X , Y):
        self.W_ = np.random.randn(X.shape[1] , np.unique(Y).size)*0.01
        self.b = np.zeros((1, np.unique(Y).size))
        Y_converted = self.one_hot(Y , np.unique(Y).size)
        for _ in range(self.n_iter):
            z = self.softmax(self.net_input(X))
            self.W_ -= np.dot(X.T , (z - Y_converted))*self.eta/X.shape[0]
            self.b -= np.sum(z - Y_converted, axis = 0)*self.eta/X.shape[0]
        return self

    def softmax(self, z):
        k = np.exp(z - np.max(z , axis = 1, keepdims=True))
        return k/np.sum(k, axis = 1, keepdims=True)

    def one_hot(self, Y, classes):
        _one_hot = np.zeros((Y.shape[0] , classes))
        _one_hot[np.arange(Y.size), Y] = 1
        return _one_hot
    
    def net_input(self, X):
        return np.dot(X , self.W_) + self.b
    
    def predict(self, X):
        z= self.net_input(X)
        return np.argmax(self.softmax(z) , axis = 1)
