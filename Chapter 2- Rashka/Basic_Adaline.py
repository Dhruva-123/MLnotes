import numpy as np
import matplotlib.pyplot as plt
class Adaline: ###This Adaline is Batch Gradient Descent Algo. There are a lot more different methods to do the exact same gradient descent that are faster and cleaner than this.
    def __init__(self, eta, n_iter , random_state):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X , Y):
        self.cost_ = []
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1] + 1)
        for _ in range(self.n_iter):
            input = self.z_function(X)
            output = self.actfunc(input)
            error = Y - output
            self.w_[1:] += self.eta*(X.T.dot(error))
            self.w_[0] += self.eta*error.sum()
            cost = (error**2).sum()/2.0
            self.cost_.append(cost)
        return self
    
    def z_function(self, X):
        return np.dot(X , self.w_[1:]) + self.w_[0]
    
    def actfunc(self, value):
        return value
    
    def predict(self, X):
        return np.where(self.actfunc(self.z_function(X)) >= 0.0 , 1, -1)
    
    def accuracy(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)*100
    
    def curve(self):
        x = [i + 1 for i in range(len(self.cost_))]
        y = self.cost_
        plt.plot(x, y)
        plt.title("Cost Curve")
        plt.xlabel("No.of iterations")
        plt.ylabel("Cost per iteration")
        plt.show()


        