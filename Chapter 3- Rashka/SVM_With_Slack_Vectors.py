import numpy as np
class SVM:
    def __init__(self, C , eta, n_iter, random_state):
        self.C = C
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, Y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1] + 1)
        for _ in range(self.n_iter):
            for xi, answer in zip(X, Y):
                output = self.net_input(xi)*answer
                if output < 1:
                    self.w_[1:] -= self.eta*(self.w_[1:] - self.C*xi*answer)
                    self.w_[0] -= self.eta*(-self.C*answer)
                else:
                    self.w_[1:] -= self.eta*(self.w_[1:])
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] if X.ndim > 1 else np.dot(self.w_[1:], X) + self.w_[0]

    def predict(self, xi):
        return np.where(self.net_input(xi) > 0, 1, -1)
    
    def accuracy(self, X , Y):
        return np.mean(self.predict(X) == Y)*100

        