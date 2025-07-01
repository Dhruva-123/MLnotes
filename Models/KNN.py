import numpy as np
class KNN:
    def __init__(self, k):
        self.k = k
    
    def neighbhour_finder(self, Data, x, method = 1):
        X = Data[: , :-1]
        arr = []
        for item in range(X.shape[0]):
            if method == 1:
                distances = np.linalg.norm(X - x, axis=1)
                arr.append([item , distance])
            else:
                distance = np.sum(np.abs(X - x) , axis = 1)
        arr.sort(key = lambda x : x[1])
        return arr[:self.k]
    
    def classifier(self, Data, X, method = 1):
        ans = []
        for x in X:
            arr = np.array(self.neighbhour_finder(Data , x, method))
            index = arr[:, 0].astype(int)
            classes = Data[index, -1].astype(int)
            ans.append(np.bincount(classes).argmax())
        return ans
    
    def regressor(self, Data, X, method = 1):
        ans = []
        for x in X:
            arr = np.array(self.neighbhour_finder(Data , x, method))
            index = arr[:, 0].astype(int)
            answers = Data[index, -1]
            ans.append(np.mean(answers))
        return ans

