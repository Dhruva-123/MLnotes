import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
class LeastSquared_Boosting:
    def __init__(self, n_estimators, learn_rate):
        self.n_estimators = n_estimators
        self.models = []
        self.learn_rate = learn_rate

    def fit(self, X, y):
        self.f_0 = np.mean(y)
        y_ = y - self.f_0
        for _ in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth = 2)
            model.fit(X, y_)
            self.models.append(model)
            y_ -= model.predict(X)*self.learn_rate

    def predict(self, X):
        answer = self.f_0
        for model in self.models:
            answer += self.learn_rate*model.predict(X)
        return answer
    
class ADABoosting:
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alphas = []
        self.models = []
    
    def fit(self, X, y):
        w = np.ones((X.shape[0]))/X.shape[0]
        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth = self.max_depth)
            model.fit(X, y, sample_weight = w)
            y_pred = model.predict(X)
            y_pred = np.where(y_pred == 0, -1, 1)
            y_temp = np.where(y == 0, -1 , 1)
            w_correct = np.sum(w*(y_temp == y_pred))
            w_incorrect = np.sum(w*(y_temp != y_pred))
            if w_incorrect > 0.5:
                break
            elif w_incorrect == 0:
                alpha = np.inf
            else:
                alpha = 0.5*np.log(w_correct/w_incorrect)
            w *= np.exp(-y_temp*y_pred*alpha)
            w /= np.sum(w)
            self.alphas.append(alpha)
            self.models.append(model)
    
    def predict(self, X):
        answer = 0
        for alpha , model in zip(self.alphas , self.models):
            k = model.predict(X)
            answer += alpha*np.where(k == 0 , -1, 1)
        return np.where(answer > 0, 1 , 0)

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)


ModelA = DecisionTreeClassifier(max_depth = 12)
ModelA.fit(X_train, y_train)
print(f"Normal Tree: {np.mean(ModelA.predict(X_test) == y_test)*100}")

ModelB = ADABoosting(n_estimators = 300 ,max_depth = 3)
ModelB.fit(X_train, y_train)
print(f"ADA Boosting: {np.mean(ModelB.predict(X_test) == y_test)*100}")        