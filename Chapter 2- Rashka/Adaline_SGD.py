import numpy as np
import matplotlib.pyplot as plt
class AdalineSGD: ###Stochastic Gradient Descent. Works cleaner than standard gradient descent.
    def __init__(self, eta, n_iter, random_state, shuffle = True):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        self.w_init = False
    
    def fit(self, X , Y):
        self.cost_ = []
        self.init_weights(X)
        for _ in range(self.n_iter):
            cost = 0.0
            if self.shuffle:
                X, Y = self.shuffle_(X, Y)
            for xi, answer in zip(X, Y):
                cost += self.update_weights(xi , answer)
            self.cost_.append(cost)
        return self
    
    def init_weights(self, X):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0, size = X.shape[1] + 1, scale = 0.01)
        self.w_init = True

    def shuffle_(self ,X , Y):
        r = self.rgen.permutation(len(Y))
        return X[r], Y[r]
    
    def update_weights(self, xi , y):
        error = y - self.net_input(xi)
        self.w_[1:] += self.eta*error*xi
        self.w_[0] += self.eta*error
        cost = (error**2)/2.0
        return cost
    
    def partial_fit(self, X, Y):
        if not self.w_init:
            self.init_weights(X)
        if Y.ravel().shape[0] > 1:
            for xi,answer in zip(X, Y):
                self.update_weights(xi , answer)
        else:
            self.update_weights(X, Y)
        return self
    
    def net_input(self, xi):
        return np.dot(xi, self.w_[1:]) + self.w_[0]
    
    def actfunc(self, net_input):
        return net_input
    
    def predict(self, X):
        return np.where(self.actfunc(self.net_input(X)) >= 0.0 , 1, -1)
    
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


        