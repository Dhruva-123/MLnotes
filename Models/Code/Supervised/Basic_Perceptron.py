###Perceptron
import numpy as np
import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self, eta, random_state, n_iter): ### n_iter = No.of times that we would like to iter over the entire dataset to train it well. Hyperparameter.
        self.eta = eta
        self.random_state = random_state
        self.n_iter = n_iter
        
    
    def fit(self, X, Y): ### X = a 2d array containing both training data and features (m * n) ; Y = an array that contains the answers for each example respectively
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01 , size = X.shape[1] + 1)
        self.errors_ = []
        for _ in range(self.n_iter): 
            errors = 0  
            for xi, answer in zip(X, Y):
                change = self.eta*(answer - self.predict(xi))
                self.w_[1:] += change*xi
                self.w_[0] += change
                errors += int(change != 0.0)### total number of errors in this iteration.
            self.errors_.append(errors)
        return self 

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return 1 if z >= 0.0 else -1
    
    def accuracy(self, X, Y):
        predictions = np.array([self.predict(xi) for xi in X])
        return np.mean(predictions == Y)*100

    def curve(self):
        x = [i + 1 for i in range(len(self.errors_))]
        y = self.errors_
        plt.plot(x, y)
        plt.title("Error Curve")
        plt.xlabel("No.of iterations")
        plt.ylabel("No.of errors")
        plt.show()
     
    