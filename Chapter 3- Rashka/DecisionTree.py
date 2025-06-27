import numpy as np
class node:
    def __init__(self, threshold = None, left = None, right = None, feature = None, value = None):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.feature = feature
        self.value = value

class DecisionTree:
    def __init__(self, min_samples, max_depth):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth
    
    def Tree(self, dataset, depth = 0):
        X = dataset[: , :-1]
        Y = dataset[: , -1]
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples and depth <= self.max_depth:
            best_split = self.best_split(dataset, num_samples, num_features)
            if best_split and best_split["info_gain"] > 0:
                left_tree = self.Tree(best_split["left"], depth + 1)
                right_tree = self.Tree(best_split["right"], depth + 1)
                return node(threshold = best_split["threshold"] , left = left_tree,right = right_tree, feature = best_split["feature"])
        value_ = self.leaf_value(Y)
        return node(value = value_)
    
    def best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')
        for feature in range(num_features):
            for sample in range(num_samples):
                threshold = dataset[sample, feature]
                left , right = self.split(dataset , threshold, feature)
                info_gain = self.info_gain(dataset, left, right)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split["threshold"] = threshold
                    best_split["left"] = left
                    best_split["right"] = right
                    best_split["info_gain"] = info_gain
                    best_split["feature"] = feature
        return best_split
    
    def split(self, dataset, threshold, feature):
        left = np.array([x for x in dataset if  x[feature] <= threshold])
        right = np.array([x for x in dataset  if x[feature] > threshold])
        return left, right
    
    def info_gain(self, dataset, left, right, mode = "entropy"):
        num_samples = dataset.shape[0]
        num_samples_left = left.shape[0]
        num_samples_right = right.shape[0]
        if mode == "entropy":
            info_gain = self.entropy(dataset[: , -1]) - (num_samples_left/num_samples)*self.entropy(left[: , -1]) - (num_samples_right/num_samples)*self.entropy(right[: , -1])
        if mode == "gini":
            info_gain = self.gini(dataset[: , -1]) - (num_samples_left/num_samples)*self.gini(left[: , -1]) - (num_samples_right/num_samples)*self.gini(right[: , -1])
        return info_gain
    
    def entropy(self, Y):
        classes = np.unique(Y)
        entropy = 0
        for item in classes:
            weight = len(Y[Y == item])/len(Y)
            entropy -= weight*np.log2(weight)
        return entropy

    def gini(self, Y):
        classes = np.unique(Y)
        gini = 1
        for item in classes:
            weight = len(Y[Y == item])/len(Y)
            gini -= weight**2
        return gini

    def leaf_value(self, Y):
        Y = list(Y)
        return max(Y , key = Y.count)
    
    def accuracy(self, dataset):
        X = dataset[: , :-1]
        Y = dataset[: , -1]
        answers = self.predict(X, self.root)
        return np.mean(answers == Y)*100
    
    def fit(self, X , Y):
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self.Tree(dataset=dataset)
    
    def predict(self, X, root = None):
        predictions = [self.make_predict(x, root) for x in X]
        return predictions
    
    def make_predict(self, x, root):
        if root.value != None:
            return root.value
        if x[root.feature] <= root.threshold:
            return self.make_predict(x , root.left)
        else:
            return self.make_predict(x , root.right)



