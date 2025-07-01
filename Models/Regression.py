import numpy as np
from itertools import combinations_with_replacement
class Regression:
    def __init__(self, eta, n_iter, random_state, lam, r = 0.5):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.lam = lam
        self.r = r

    def LinearRegression(self, X, Y, type = "Ridge"):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1] + 1)
        for _ in range(self.n_iter):
            dw = (-2 /X.shape[0]) * np.dot(X.T, (Y - self.predict(X)))
            db = (-2 /X.shape[0]) * np.sum(Y - self.predict(X))
            if type == "Ridge":
                dw += 2*self.lam*self.w_[1:]
            else:
                if type == "Lasso":
                    dw += self.lam*np.sign(self.w_[1:])
                else:
                    dw += self.r*(2*self.lam*self.w_[1:]) + (1-self.r)*self.lam*np.sign(self.w_[1:])
            self.w_[1:] -= self.eta*dw
            self.w_[0] -= self.eta*db

    def predict(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def PolyRegression(self, X, Y, degree ,type = "Ridge"):
        def gen(X, degree):
            n_samples, n_features = X.shape
            features = [np.ones(n_samples)]
            for d in range(1, degree + 1):
                for c in combinations_with_replacement(range(n_features), d):
                    new_term = np.ones(n_samples)
                    for comb in c:
                        new_term *= X[:, comb]
                    features.append(new_term)
            return np.vstack(features).T
        self.LinearRegression(gen(X, degree), Y, type)
    
    def rmse(self, X, Y):
        z = self.predict(X)
        return np.sqrt(np.mean((z - Y)**2))
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return (1 - ss_res / ss_tot)*100

        


        