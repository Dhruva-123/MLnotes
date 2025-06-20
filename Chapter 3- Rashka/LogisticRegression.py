import numpy as np
class LogisticRegression:
    def __init__(self, eta, random_state, n_iter):
        self.eta = eta
        self.random_state = random_state
        self.n_iter = n_iter
    
    def fit(self, X, Y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01 , size = 1 + X.shape[1])
        cost_ = []
        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = Y - output
            self.w_[1:] += X.T.dot(errors)*self.eta
            self.w_[0] += self.eta*errors.sum()
            cost = cost = -Y.dot(np.log(output.clip(1e-10, 1 - 1e-10))) - (1 - Y).dot(np.log((1 - output).clip(1e-10, 1 - 1e-10)))
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
    
