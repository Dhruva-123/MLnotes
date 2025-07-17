import numpy as np

class node:
    def __init__(self, threshold=None, left=None, right=None, feature=None, value=None, samples=None, gini=None):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.feature = feature
        self.value = value
        self.samples = samples
        self.gini = gini

class DecisionTree:
    def __init__(self, min_samples, max_depth, ccp_alpha=0.0):
        self.root = None
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha

    def Tree(self, dataset, depth=0):
        X = dataset[:, :-1]
        Y = dataset[:, -1]
        num_samples, num_features = X.shape
        gini_value = self.gini(Y)

        if num_samples >= self.min_samples and depth <= self.max_depth:
            best_split = self.best_split(dataset, num_samples, num_features)
            if best_split and best_split["info_gain"] > 0:
                left_tree = self.Tree(best_split["left"], depth + 1)
                right_tree = self.Tree(best_split["right"], depth + 1)
                return node(threshold=best_split["threshold"], left=left_tree, right=right_tree,
                            feature=best_split["feature"], samples=num_samples, gini=gini_value)
        value_ = self.leaf_value(Y)
        return node(value=value_, samples=num_samples, gini=gini_value)

    def prune(self, root):
        if root is None or root.value is not None:
            return root
        root.left = self.prune(root.left)
        root.right = self.prune(root.right)

        if root.left.value is not None and root.right.value is not None:
            error_before = root.left.gini * root.left.samples + root.right.gini * root.right.samples
            error_after = root.gini * root.samples
            if error_after + self.ccp_alpha <= error_before:
                return root
            else:
                root.left = None
                root.right = None
                root.feature = None
                root.threshold = None
                root.value = self.leaf_value_from_gini(root)
        return root

    def leaf_value_from_gini(self, node):
        # Placeholder fallback if gini only is available; can be replaced with actual majority class
        return 0

    def best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')
        for feature in range(num_features):
            thresholds = np.unique(dataset[:, feature])
            for threshold in thresholds:
                left, right = self.split(dataset, threshold, feature)
                info_gain = self.info_gain(dataset, left, right)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split = {
                        "threshold": threshold,
                        "left": left,
                        "right": right,
                        "info_gain": info_gain,
                        "feature": feature
                    }
        return best_split

    def split(self, dataset, threshold, feature):
        left = np.array([x for x in dataset if x[feature] <= threshold])
        right = np.array([x for x in dataset if x[feature] > threshold])
        return left, right

    def info_gain(self, dataset, left, right, mode="gini"):
        num_samples = dataset.shape[0]
        num_samples_left = left.shape[0]
        num_samples_right = right.shape[0]
        if num_samples_left == 0 or num_samples_right == 0:
            return 0
        if mode == "entropy":
            info_gain = self.entropy(dataset[:, -1]) - (num_samples_left / num_samples) * self.entropy(left[:, -1]) - (num_samples_right / num_samples) * self.entropy(right[:, -1])
        else:
            info_gain = self.gini(dataset[:, -1]) - (num_samples_left / num_samples) * self.gini(left[:, -1]) - (num_samples_right / num_samples) * self.gini(right[:, -1])
        return info_gain

    def entropy(self, Y):
        classes = np.unique(Y)
        entropy = 0
        for item in classes:
            weight = len(Y[Y == item]) / len(Y)
            entropy -= weight * np.log2(weight)
        return entropy

    def gini(self, Y):
        classes = np.unique(Y)
        gini = 1
        for item in classes:
            weight = len(Y[Y == item]) / len(Y)
            gini -= weight ** 2
        return gini

    def leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def accuracy(self, X, Y):
        answers = self.predict(X, self.root)
        return np.mean(answers == Y) * 100

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self.Tree(dataset=dataset)
        self.root = self.prune(self.root)

    def predict(self, X, root=None):
        if root is None:
            root = self.root
        predictions = [self.make_predict(x, root) for x in X]
        return predictions

    def make_predict(self, x, root):
        if root.value is not None:
            return root.value
        if x[root.feature] <= root.threshold:
            return self.make_predict(x, root.left)
        else:
            return self.make_predict(x, root.right)


from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

data = load_iris()
X = data.data
Y = data.target
kf = StratifiedKFold(shuffle=True, n_splits=4, random_state=42)

for train, test in kf.split(X, Y):
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]

DT = DecisionTree(min_samples=4, max_depth=10, ccp_alpha=0)
DT.fit(X_train, Y_train)
print(DT.accuracy(X_test, Y_test))
