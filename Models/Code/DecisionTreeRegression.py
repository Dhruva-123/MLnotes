import numpy as np
class Node:
    def __init__(self, left = None, right = None, feature = None,threshold = None, var = None, value = None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.var = var
        self.value = value

class DecisionTreeRegrression:
    def __init__(self, max_depth = 5, min_samples = 2):
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
    
    def Tree(self, Data, depth):
        X , Y = Data[: , :-1] , Data[:, -1]
        samples, features = X.shape
        if self.max_depth >= depth and self.min_samples <= samples:
            best_split = self.best_split(Data, samples, features)
            if best_split["var"] > 0:
                left = self.Tree(best_split["left"], depth + 1)
                right = self.Tree(best_split["right"] , depth + 1)
                return Node(left = left, right = right, threshold = best_split["threshold"], var = best_split["var"], feature = best_split["feature"])
        value = self.value_cal(Y)   
        return Node(value = value)
    
    def best_split(self, Data, samples, features):
        best_split = {}
        var_max = -float('inf')
        for feature in range(features):
            for sample in range(samples):
                threshold = Data[sample, feature]
                left, right = self.split(Data, threshold, feature)
                if len(left) == 0 or len(right) == 0:
                    continue
                var = self.var_red(Data, left, right)
                if var_max > var:
                    var_max = var
                    best_split["threshold"] = threshold
                    best_split["left"] = left
                    best_split["right"] = right
                    best_split["var"] = var
                    best_split["feature"] = feature
        return best_split
    
    def split(self, Data, threshold, feature):
        left = np.array([x for x in Data if x[feature] <= threshold])
        right = np.array([x for x in Data if x[feature] > threshold])
        return left , right
    
    def var_red(self, Data, left, right):
        y_Data = Data[:, -1]
        y_left = left[:, -1]
        y_right = right[:, -1]
        weight_l = len(y_left)/len(y_Data)
        weight_r = len(y_right)/len(y_Data)
        reduction = np.var(y_Data) - weight_l*np.var(y_left) - weight_r*np.var(y_right)
        return reduction

    def value_cal(self, Y):
        return np.mean(Y)
    
    def fit(self, X, Y, depth = 10):
        Data = np.concatenate((X , Y), axis = 1)
        self.root = self.Tree(Data, depth)
        
    def predict(self, X):
        return np.array([self.make_prediction(x, self.root) for x in X])
    
    def make_prediction(self,x, root):
        if root.value != None:
            return root.value
        if x[root.feature] <= root.threshold:
            return self.make_prediction(x, root.left)
        else:
            return self.make_prediction(x, root.right)
    
    def rmse(self, X , Y):
        predictions = self.predict(X)
        return np.sqrt(np.sum((predictions - Y)**2))/X.shape[0]

        


    


        