import numpy as np

class Gaussian_Mixture_model_EM:
    def __init__(self, n_clusters = 2, n_iter = 100, tolerance = 1e-4):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.tolerance = tolerance

    def init_parameters(self, X):
        self.n, self.d = X.shape 
        self.mu = X[np.random.choice(self.n, self.n_clusters, replace= False)]
        self.sigma = np.array([np.eye(self.d) for _ in range(self.n_clusters)]) 
        self.w = np.ones(self.n_clusters)/self.n_clusters

    def multivariate_gaussian(self, X, sigma, mu):
        n, d = X.shape
        sigma_det = np.linalg.det(sigma)
        sigma_inv = np.linalg.inv(sigma)
        norm_const = 1.0 / (np.power(2 * np.pi, d/2) * np.sqrt(sigma_det))
        
        diff = X - mu
        exponent = -0.5 * np.sum(diff @ sigma_inv * diff, axis=1)
        return norm_const * np.exp(exponent)


    def calculate_responsibility(self, X):
        gamma = np.zeros((self.n, self.n_clusters))
        for j in range(self.n_clusters):
            gamma[: , j] = self.w[j]*self.multivariate_gaussian(X, self.sigma[j], self.mu[j])

        gamma /= gamma.sum(axis = 1, keepdims = True)
        return gamma
    
    def update_parameters(self, X, gamma):
        Nk = gamma.sum(axis=0)
        
        # Update means
        self.mu = (gamma.T @ X) / Nk[:, np.newaxis]
        
        # Update covariances
        for j in range(self.n_clusters):
            diff = X - self.mu[j]
            self.sigma[j] = (gamma[:, j][:, np.newaxis] * diff).T @ diff / Nk[j]
        
        # Update weights
        self.w = Nk / self.n
    
    def log_likelihood(self, X):
        ll = np.zeros(self.n)
        for j in range(self.n_clusters):
            ll += self.w[j] * self.multivariate_gaussian(X, self.sigma[j],  self.mu[j])
        return np.sum(np.log(ll))
    
    def fit(self, X):
        self.init_parameters(X)
        log_likelihood_old = None
        for i in range(self.n_iter):
            gamma = self.calculate_responsibility(X)
            self.update_parameters(X, gamma)
            log_likelihood = self.log_likelihood(X)
            if log_likelihood_old is not None and np.abs(log_likelihood - log_likelihood_old) < self.tolerance:
                print(f"Converged at iteration : {i + 1} ")
                break
            log_likelihood_old = log_likelihood
        self.responsibilities = gamma
        return self
    
    def predict_proba(self, X):
        return self.calculate_responsibility(X)
    
    def predict(self, X):
        return np.argmax(self.calculate_responsibility(X), axis = 1)




if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    
    # Generate synthetic 2D data
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
    
    # Fit GMM
    gmm = Gaussian_Mixture_model_EM(n_clusters=3)
    gmm.fit(X)
    
    # Soft assignments
    gamma = gmm.predict_proba(X)
    
    # Hard assignments
    labels = gmm.predict(X)
    
    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(gmm.mu[:, 0], gmm.mu[:, 1], c='red', marker='x', s=100)
    plt.title("GMM Clustering")
    plt.show()