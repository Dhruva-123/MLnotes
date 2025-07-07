import numpy as np
class NaiveBayes:
    def fit(self, X, y):
        return NotImplementedError()
    
    def joint_log_likelihood(self, X):
        return NotImplementedError()
    
    def predict(self, X):
        table = self.joint_log_likelihood(X)
        return self.classes_[np.argmax(table, axis = 1)]
    
class Multinomial(NaiveBayes):
    def __init__(self, alpha = 1):
            self.alpha = alpha
        
    def fit(self, X , y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        self.class_matrix = np.zeros((n_classes))
        self.feature_matrix = np.zeros((n_classes , n_features))
        for index, class_ in enumerate(self.classes_):
            X_ = X[y == class_]
            self.class_matrix[index] = X_.shape[0]
            self.feature_matrix[index] = np.sum(X_, axis = 0)
        self.class_log_prior = np.log(self.class_matrix/n_samples)
        self.class_count = np.sum(self.feature_matrix + self.alpha, axis = 1).reshape((-1, 1))
        self.feature_log_likelihood = np.log((self.feature_matrix + self.alpha)/self.class_count)
        
    def joint_log_likelihood(self, X):
        return X @ self.feature_log_likelihood.T + self.class_log_prior

class Gaussian(NaiveBayes):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.means = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.prior = np.zeros((n_classes))
        for index, item in enumerate(self.classes_):
            X_ = X[y == item]
            self.means[index] = np.mean(X_, axis = 0)
            self.var[index] = np.var(X_, axis = 0) + 1e-9
            self.prior[index] = X_.shape[0]/X.shape[0] 

    def joint_log_likelihood(self, X):
        classes = np.unique(self.classes_)
        log_posterior = np.zeros((X.shape[0], len(classes)))
        for index in classes:
            mean = self.means[index]
            var = self.var[index]
            prior = self.prior[index]
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * var))
            log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_posterior[:, index] = prior + log_prob
        return log_posterior

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
accuracy = []
for train_index, test_index in skf.split(X, y):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    model = Multinomial()
    model.fit(X_train , y_train)
    accuracy.append(np.mean(model.predict(X_test) == y_test)*100)
print(np.mean(accuracy))



