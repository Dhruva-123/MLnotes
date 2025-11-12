
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


### EM Gaussian Models

To me, Gaussian EM models are little different from fuzzy C-means. In fuzzy C-means you have normal calculations of what a cluster is and here, we use gaussian models. First of all, we assume each cluster is a gaussian distribution. To properly define a gaussian distribution, we need to have $\mu_j$ (where $\mu$ is the mean of the entire distribution and $j$ is the ID of the cluster) and the $\Sigma_j$ (The covariance matrix of the given $j^{th}$ cluster). The covariance matrix gives us a general idea of how off we are and how right we are. As we have discussed in Fuzzy C-Means, $w_{ij}$ stands for the probability of the $i^{th}$ sample being a part of the $j^{th}$ cluster. Here,  we will sum that weight across all the samples which leads to this : $w_j$. What this tells us is, among all the distributions, what is the probability of a random $x_i$ being in the $j^{th}$ cluster. In these models, $w_j$ is also a parameter. 

NOTE:

Whenever we have $w_j$, we have to make sure that the sum of all $w_j$ must be equal to 1 or else the entire probability fails.

With these parameters we have established, we can cluster them together into a set like so:

$θ_j$​=($w_j​$,$μ_j$​,$Σ_j$​)

Now, we will see the gaussian density probability function: (This is multivariate because we have more than 1 feature)

$$
\mathcal{N}(x_i \mid \mu_j, \Sigma_j) =
\frac{1}{(2 \pi)^{d/2} \, |\Sigma_j|^{1/2}}
\exp \Bigg( -\frac{1}{2} (x_i - \mu_j)^T \, \Sigma_j^{-1} \, (x_i - \mu_j) \Bigg )
$$
This is the gaussian density probability function. The $d$ you see here is the number of features in the data that we provided. 

Here is how to calculate the probability array of a sample using that gaussian multivariate formula:

$$
p(x_i \mid \Theta) = \sum_{j=1}^{K} w_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)
$$
All of this is the theory behind how we calculate the probability for each sample. In practice, we have to morph the formulae a bit in order to make our life easier. Therefore, here is the practical algorithm for this:

#### **1. Initialization**

You start by **randomly initializing**:

- $\mu_j$​: the mean vector of each Gaussian
    
- $\Sigma_j$​: the covariance matrix
    
- $w_j$: the weight (mixing coefficient) — the prior probability of each cluster
    

Formally:

$$\sum_{j=1}^{K} w_j = 1$$

---

#### **2. E-step (Expectation)**

Using these initial parameters, you compute the **responsibility** — that is, how much each cluster “claims” a data point.

For each data point $x_i$​ and cluster $j$:

$$
\gamma_{ij} = \frac{w_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
                {\sum_{l=1}^{K} w_l \, \mathcal{N}(x_i \mid \mu_l, \Sigma_l)}
$$

This gives you a **soft assignment**, very similar in spirit to **Fuzzy C-Means membership weights** — except these are now **probabilities** derived from the Gaussian likelihoods.

---

#### **3. M-step (Maximization)**

Now, update the parameters using those responsibilities.

New weights:

$$
w_j = \frac{1}{n} \sum_{i=1}^{n} \gamma_{ij}
$$

New means:

$$
\mu_j = \frac{\sum_{i=1}^{n} \gamma_{ij} \, x_i}{\sum_{i=1}^{n} \gamma_{ij}}
$$
New covariance matrices:

$$
\Sigma_j = \frac{\sum_{i=1}^{n} \gamma_{ij} \, (x_i - \mu_j)(x_i - \mu_j)^T}{\sum_{i=1}^{n} \gamma_{ij}}
$$

---

#### **4. Evaluate the Log-Likelihood**

You compute the **log-likelihood** of the data given the parameters:

$$
\mathcal{L} = \sum_{i=1}^{n} \log \left( \sum_{j=1}^{K} w_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j) \right)
$$

You check how much it improves.  
If the improvement is tiny (below a tolerance), you stop


There are a few points to note here:

- The point of this model is to return the $\gamma_{ij}$ matrix to the user. That is what contains the probability needed for the user. Unlike fuzzy C-Means, we don't return $w_j$.
- The log likelihood function written at the bottom has no use in the training of the model. We can use the data given by it to draw a graph btw the number of iterations and the log-Likelihood to find out the optimal number of iterations. The log likelihood is basically the error.

We will be writing the python code to this.
