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
        if num_samples > self.min_samples and depth < self.max_depth:
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
            thresholds = np.unique(dataset[:, feature])
            thresholds = np.random.choice(thresholds, size = min(10 , len(thresholds)), replace = False)
            for threshold in thresholds:
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
        mask = dataset[: , feature] <= threshold
        left = dataset[mask]
        right = dataset[~mask]
        return left, right
    
    def info_gain(self, dataset, left, right, mode = "entropy"):
        num_samples = dataset.shape[0]
        num_samples_left = left.shape[0]
        num_samples_right = right.shape[0]
        left = np.atleast_2d(left)
        right = np.atleast_2d(right)
        if left.shape[1] == 0 or right.shape[1] == 0:
            return 0  # skip this split candidate
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
    
    def accuracy(self, X, Y):
        answers = self.predict(X, self.root)
        return np.mean(answers == Y)*100
    
    def fit(self, X , Y):
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self.Tree(dataset=dataset)
    
    def predict(self, X, root = None):
        if root == None:
            root = self.root
        predictions = np.array([self.make_predict(x, root) for x in X])
        return predictions
    
    def make_predict(self, x, root):
        if root.value != None:
            return root.value
        if x[root.feature] <= root.threshold:
            return self.make_predict(x , root.left)
        else:
            return self.make_predict(x , root.right)

from sklearn.tree import DecisionTreeClassifier
class ADABoosting:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []
        self.errors = []
    
    def fit(self, X , y):
        y = np.where(y == 0, -1, 1)
        n_samples = X.shape[0]
        weights = np.ones(n_samples)/n_samples
        for _ in range(self.n_estimators):
            #shuffled_indicies = np.random.choice(n_samples , size = n_samples , p = weights)
            #X_ = X[shuffled_indicies]
            #y_ = y[shuffled_indicies]
            model = DecisionTreeClassifier(max_depth = 1)
            model.fit(X, y, sample_weight = weights)
            y_pred = np.array(model.predict(X))
            incorrect = (y_pred != y).astype(int)
            incorrect = incorrect.ravel()
            error = np.dot(weights, incorrect)/np.sum(weights)
            error = np.clip(error, 1e-10, 1 - 1e-10)  # Avoid extreme values
            if error > 0.5:
                print(f"You've got a bad learner... flipping sign")
                y_pred = -1*y_pred
            self.errors.append(error)
            alpha = 0.5*np.log((1 - error)/error)
            alpha = np.clip(alpha, 0 , 2)
            weights *= np.exp(-y*y_pred*alpha)
            weights /= np.sum(weights)               
            weights = weights.ravel()
            self.alphas.append(alpha)
            self.models.append(model)

    def predict(self, X):
        prediction = np.zeros((X.shape[0]))
        for alpha, model in zip(self.alphas , self.models):
            prediction += alpha*np.array(model.predict(X))
        return np.where(prediction > 0 , 1 , 0)

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=2000, noise=0.5)

SKF = StratifiedKFold(n_splits = 5)
for train, test in SKF.split(X , y):
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]
iterations = 1000
modelA = ADABoosting(n_estimators = iterations)
modelA.fit(X_train, y_train)
x_axis = np.array([i for i in range(1, iterations + 1)])
y_axis = np.array(modelA.errors)
plt.plot(x_axis, y_axis)
plt.xlabel("interation count")
plt.ylabel("error")
plt.show()
print(f"ADABoosting : {np.mean(modelA.predict(X_test) == y_test)*100}")

modelB = DecisionTreeClassifier(max_depth = 10, min_samples_split = 4)
modelB.fit(X_train, y_train)
print(f"SKlearn Tree : {np.mean(modelB.predict(X_test) == y_test)*100}")

modelC = DecisionTree(max_depth = 10 , min_samples = 4)
modelC.fit(X_train , y_train)
print(f"My Tree: {np.mean(modelC.predict(X_test) == y_test)*100}")

