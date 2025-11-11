
In this section, we will discuss about soft clustering. In hard clusters, our classification or clustering either yes or no. You are either a part of a given cluster or you are not. but with soft cluster, we are using probability on a scale of 0-1. This is quite like naive bayes.  We will be discussing soft clustering. Here are the top models we will be talking about:

- Fuzzy C-means
- Gaussian mixtures
- AIC and BIC as performance metrics
- Bayesian Gaussian mixtures
- Generative Gaussian mixtures.

The thing with this soft clustering is, this is amazing when we are trying to feed these outputs of the soft clustering models into another neural network for other purposes. This gives us a wide range of knowledge on a particular point given not just a single character defining where it is clustered as. This gives a large amount of data. Therefore, I think, we should use this when we are trying to preprocess data before feeding into any model.


### Fuzzy C-Means

Fuzzy c-means is an algorithm that is created on the basis of the standard k-means algorithm. This contains the probability of each point in the given sample, having an array of shape (no.of clusters, 1). Here, clusters are overlapping. Note that the sum of all the values inside the array should be equal to 1.

```
from skfuzzy.cluster import cmeans
Ws = []
pcs = [] 
for m in np.linspace(1.05, 1.5, 5): 
fc, W, _, _, _, _, pc = cmeans(X.T, c=10, m=m, error=1e-6, maxiter=20000, seed=1000) 
Ws.append(W) 
pcs.append(pc)
```
That would be the sklearn implementation of the model.


There is some math behind this model that we all need to understand to the core. Here is how the model works (mathematically):

- This assigns probability values to each sample on what cluster the sample belongs. In order to do that, we first have a loss function that we are trying to minimize:

$J_m = \sum_{i=1}^{N} \sum_{j=1}^{C} u_{ij}^m \, \|x_i - c_j\|^2$

Where:

- $u_{ij}$​ = membership of point $x_i$ in cluster j
    
- $m > 1$ = fuzziness parameter (usually 2). Note that if m = 1, it's just standard hard clustering and not soft clustering.
    
- $c_j$​ = centroid of cluster j

In order to recalculate the centroids like in K-means, we have a formula for centroids:

**Centroids**

$$c_j = \frac{\sum_{i=1}^{N} u_{ij}^m \, x_i}{\sum_{i=1}^{N} u_{ij}^m}$$

Where $C_j$ is the centroid of the $j^{th}$ center. 

**Membership ($U_{ij}$)**

	$$
u_{ij} = \frac{1}{
  \displaystyle \sum_{k=1}^{C} 
    \left(
      \frac{\lVert x_i - c_j \rVert}{\lVert x_i - c_k \rVert}
    \right)^{\frac{2}{m-1}}
}
$$

We need to incorperate these formulas in python. In the centroids, we need to calculate the inverse of matrix $u_{ij}$ and then matrix multiply with $X$ because we need to manage the dimensions well. In practice, we add a little error for membership calculations because we cannot afford to divide by zero.

Except that, This is exactly the same as K-means.