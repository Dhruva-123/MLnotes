
### 🔹 **Definition & Intuition**

- **KNN** is a **lazy learning** algorithm: no training phase, just memorization.
    
- Prediction is made by finding the **K nearest data points** to a new input and using them to:
    
    - **Classify** (majority vote)
        
    - **Regress** (average target values)

### 🔸 **Core Concepts**

| Concept         | Description                                             |
| --------------- | ------------------------------------------------------- |
| `k`             | Number of neighbors to consider                         |
| Distance Metric | How similarity is computed (Euclidean, Manhattan, etc.) |
| Decision Rule   | Majority vote (classifier), average (regressor)         |

### 🧠 **Distance Metrics**

Assume `x` is a query point and `X` is dataset (`n_samples x n_features`):

|Name|Formula|Vectorized Code|
|---|---|---|
|**Euclidean**|√∑(xᵢ - yᵢ)²|`np.sqrt(np.sum((X - x)**2, axis=1))`|
|**Manhattan**|∑|xᵢ - yᵢ|
|**Minkowski(p)**|(∑|xᵢ - yᵢ|
|**Cosine**|1 - (x·y / ‖x‖‖y‖)|see below ⬇|

x_norm = np.linalg.norm(x)
X_norms = np.linalg.norm(X, axis=1)
cosine_dist = 1 - (X @ x) / (X_norms * x_norm + 1e-8)

CODE:
class KNN:
    def __init__(self, k):                # store k
        self.k = k

    def _distance(self, X, x, method=1):  # compute distance
        if method == 1:
            return np.sqrt(np.sum((X - x) ** 2, axis=1))
        elif method == 2:
            return np.sum(np.abs(X - x), axis=1)
        else:
            raise ValueError("Unknown method")

    def neighbhour_finder(self, Data, x, method=1):
        X = Data[:, :-1]
        distances = self._distance(X, x, method)
        indexed = np.array(list(enumerate(distances)))
        indexed = indexed[indexed[:, 1].argsort()]
        return indexed[:self.k]

    def classifier(self, Data, X, method=1):
        ans = []
        for x in X:
            neighbors = self.neighbhour_finder(Data, x, method)
            indices = neighbors[:, 0].astype(int)
            classes = Data[indices, -1].astype(int)
            ans.append(np.bincount(classes).argmax())
        return ans

    def regressor(self, Data, X, method=1):
        ans = []
        for x in X:
            neighbors = self.neighbhour_finder(Data, x, method)
            indices = neighbors[:, 0].astype(int)
            values = Data[indices, -1]
            ans.append(np.mean(values))
        return ans


### 🎯 **Strategic Usage Notes**

- **K = odd** for binary classification to avoid ties.
    
- **K too small → high variance**, **K too large → high bias**
    
- **Distance metric matters.** Scale features (e.g. with `StandardScaler`), or distances become meaningless.
    
- Use `KD-Tree` or `BallTree` for speed when dataset is large.


### ⚠️ **Common Pitfalls**

- Not converting neighbor indices to `int` → breaks slicing.
    
- Using unscaled features → distance becomes distorted.
    
- High-dimensional data → distances lose meaning (curse of dimensionality).

### Extension Ideas

- Add `fit()` + `predict()` to make it `scikit-learn` compatible.
    
- Add `weights` = inverse distance for better predictions.
    
- Vectorize fully to eliminate all loops.


In brute-force KNN:

- For each query point, you compute distance to **all** training points → **O(n)** time per query.
    
- For `n` queries → **O(n²)** total.
    

That’s fine for tiny datasets — but **breaks at scale**.

So we use **spatial indexing trees** to reduce the search space.

---

## 🔺 KD-Tree (K-Dimensional Tree)

### ✅ What it is:

- A binary tree that recursively splits the data along the **feature axes** (e.g. x, y, z...).
    
- At each node, it chooses a dimension and splits the dataset at the median.
    

### 🧠 How it helps:

- Narrows down the nearest neighbors using geometric regions.
    
- **Query time ≈ O(log n)** for low-dimensional data (`<30 dims`).
    

### ⚠️ When it fails:

- In **high dimensions**, the tree becomes ineffective → query time back to brute-force.
    

---

## 🟣 Ball Tree

### ✅ What it is:

- Generalization of KD-tree. Instead of axis-aligned splits, it groups points in **hyperspheres** (balls).
    
- Each node contains a **ball with a center and radius** that encloses a subset of points.
    

### 🧠 When to use:

- **Works better than KD-tree in higher dimensions**
    
- Efficient for Euclidean, Manhattan, and custom distance metrics
    

---

## ⚔️ KD-Tree vs Ball Tree

|Feature|KD-Tree|Ball Tree|
|---|---|---|
|Splits|Axis-aligned|Hypersphere|
|Best for|Low dims (<30)|Mid/high dims|
|Speed|Very fast in low-D|More stable in high-D|
|Distance types|L2, L1|L2, L1, custom|

---

## 🔧 In Practice

In `scikit-learn`, you don’t need to implement these manually:


`from sklearn.neighbors import KNeighborsClassifier  knn = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')  # or 'ball_tree' or 'auto'`

Use:

- `'kd_tree'` for low dimensions
    
- `'ball_tree'` if dims > 30
    
- `'brute'` for small data
    
- `'auto'` → picks best one automatically