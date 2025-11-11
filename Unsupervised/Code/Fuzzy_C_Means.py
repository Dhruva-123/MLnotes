import numpy as np

class FuzzyCMeans:
    def __init__(self, n_clusters=2, m=2, max_iter=100, error=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error

    def initialize_membership(self, n_samples):
        # Random membership matrix, sum of memberships per sample = 1
        U = np.random.rand(n_samples, self.n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)
        return U

    def update_centroids(self, X, U):
        um = U ** self.m
        centroids = (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)
        return centroids

    def update_membership(self, X, centroids):
        dist = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # Avoid division by zero
        dist = np.fmax(dist, 1e-10)
        power = 2 / (self.m - 1)
        inv_dist = dist ** (-power)
        U_new = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        return U_new

    def fit(self, X):
        n_samples = X.shape[0]
        U = self.initialize_membership(n_samples)
        
        for i in range(self.max_iter):
            centroids = self.update_centroids(X, U)
            U_new = self.update_membership(X, centroids)
            # Convergence check
            if np.linalg.norm(U_new - U) < self.error:
                break
            U = U_new
        
        self.centroids = centroids
        self.U = U
        return self

    def predict(self, X):
        # Assign to the cluster with highest membership
        dist = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        power = 2 / (self.m - 1)
        inv_dist = dist ** (-power)
        U_new = inv_dist / np.sum(inv_dist, axis=1, keepdims=True)
        return np.argmax(U_new, axis=1)


# Example usage
if __name__ == "__main__":
    # Sample data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    fcm = FuzzyCMeans(n_clusters=3)
    fcm.fit(X)
    labels = fcm.predict(X)

    print("Centroids:\n", fcm.centroids)
    print("Labels:\n", labels)
