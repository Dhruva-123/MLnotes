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
    def __init__(self, alpha):
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
        return X @ self.feature_log_likelihood + self.class_log_prior
            