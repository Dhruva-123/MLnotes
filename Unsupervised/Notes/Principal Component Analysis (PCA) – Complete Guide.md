

## **1. Intuition**

Principal Component Analysis (PCA) is a **dimensionality reduction technique** used in unsupervised learning.

Imagine you have a dataset with many correlated features. Some of these features may not provide new information—they’re redundant or highly correlated. PCA helps you:

1. Reduce the number of features (dimensions) while retaining most of the **variance** (information).
    
2. Identify the **principal directions** along which the data varies the most.
    
3. Transform the data into a new coordinate system defined by these directions (principal components).
    

Think of it as **rotating your axes** to align with the directions where the data spreads the most, and then dropping the axes that contribute very little variance.

---

## **2. Mathematical Overview**

Suppose your dataset is X of shape (n_samples,n_features)

### **Step 1: Center the data**

$Xcentered=X−μ$

**Why?** PCA assumes data is centered around 0 so that variance is meaningful.

---

### **Step 2: Covariance matrix**

Compute covariance to understand how features vary together:

$$
\text{Cov} = \frac{1}{n-1} X_{\text{centered}}^T X_{\text{centered}}
$$

- Covariance between feature i and j: high value → they vary together.
    
- Diagonal entries: variance of each feature.
    

---

### **Step 3: Eigen decomposition**

We want the **principal components**.

- Solve:
    

$$
\text{Cov} \, v = \lambda v
$$
- v = eigenvector → direction of a principal component
    
- $\lambda$ = eigenvalue → variance along that component
    

Sort eigenvectors by descending eigenvalues. Higher eigenvalue → more variance captured.

---

### **Step 4: Select top components**

- Pick top k eigenvectors corresponding to largest eigenvalues.
    
- Form a matrix $V_k$​ of shape (nfeatures,k).
    

---

### **Step 5: Project data**

Transform data to reduced dimensions:

$X_{\text{reduced}} = X_{\text{centered}} \cdot V_k$

- $X_{\text{reduced}}$ is shape $(n_{\text{samples}}, k)$ 
    
- Now your data lives in k-dimensional space with maximum variance preserved.
    

---

### **Explained variance**

$\text{Explained Variance Ratio} = \frac{\lambda_i}{\sum_j \lambda_j}$

- Tells you how much variance each principal component captures.
    
- Used to select the number of components (e.g., keep 95% of variance).


## **4. Practical Applications**

1. **Feature reduction for ML** – reduces computation and noise for downstream models.
    
2. **Visualization** – project high-dimensional data to 2D or 3D.
    
3. **Noise reduction** – drop low-variance components, keeping only the “signal.”
    
4. **Preprocessing for clustering** – reduce dimensionality before k-means, DBSCAN, or Gaussian mixtures.
    
5. **Anomaly detection** – outliers often appear far in PCA space.
    

---

## **5. MLOps / Production Considerations**

- **Save/load PCA** using joblib or pickle.
    
- Apply the same PCA transformation on new incoming data (`pca.transform(new_X)`).
    
- Ensure consistent scaling (train/test).
    
- Monitor explained variance in production if features change over time.
    
- Optional: dynamically choose `n_components` using variance threshold (e.g., 95%).
    

---

## **6. Limitations**

- Linear method → cannot capture non-linear relationships.
    
- Sensitive to feature scaling.
    
- Principal components are linear combinations → hard to interpret sometimes.
    
- Kernel PCA or autoencoders can handle non-linear data if needed.