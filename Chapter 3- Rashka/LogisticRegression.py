import numpy as np
class LogisticRegression:
    def __init__(self, eta, random_state, n_iter):
        self.eta = eta
        self.random_state = random_state
        self.n_iter = n_iter
    
    def fit(self, X, Y):
        cost_ = []
        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = 1 - output
            self.w_[1:] += X.T.dot(errors)*self.eta
            self.w_[0] += self.eta*errors.sum()
            cost = -Y.dot(np.log(output)) - (1 - Y).dot(np.log(1 - output))
            cost_.append(cost)
        return self
    
    def activation(self, z):
        return 1/(1 + (np.exp(-np.clip(z, -250, 250))))
    
    def net_input(self, X):
        return np.dot(X , self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
    def accuracy(self, X, Y):
        return np.mean(self.predict(X) == Y)*100
    
