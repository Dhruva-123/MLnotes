import numpy as np
import matplotlib.pyplot as plt

class kmeanspp:
    def __init__(self, k = 3, n_iter = 200):
        self.k = k
        self.n_iter = n_iter
        self.centroids = []

    def init_centroids(self, X):
        centroid = X[np.random.choice(len(X))]
        self.centroids.append(centroid)
        for _ in range(self.k - 1):
            distances = np.array([min([np.sum((x - c)**2) for c in self.centroids]) for x in X])
            probs = distances / np.sum(distances)
            next_idx = np.random.choice(len(X), p=probs)
            self.centroids.append(X[next_idx])
        self.centroids = np.array(self.centroids)  
        
        
    def euclidean_distance(self, point, random_centers): ## This is simple Euclidean distance.
        return np.sqrt(np.sum((random_centers - point)**2, axis = 1))

    def fit(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.init_centroids(X)
        for _ in range(self.n_iter):
            y = []
            for point in X:
                distance = self.euclidean_distance(point, self.centroids)
                group = np.argmin(distance)
                y.append(group)
            grouped_indices = []
            for i in range(self.k):
                grouped_indices.append(np.argwhere(y == i))
            calculated_centroids = []
            for group in grouped_indices:
                calculated_centroids.append(np.mean(X[group.flatten()], axis = 0))
            calculated_centroids = np.array(calculated_centroids)
            if np.max(np.abs(calculated_centroids - self.centroids)) >= 0.001:
                self.centroids = calculated_centroids
        return y
    
def generate_clusters(n_points_per_cluster=50, centers=None, std=5):
    if centers is None:
        centers = [[10, 10], [50, 50], [80, 20]]  # default cluster centers
    X = []
    for center in centers:
        cluster_points = np.random.randn(n_points_per_cluster, 2) * std + center
        X.append(cluster_points)
    X = np.vstack(X)
    return X

# Example
X_test = generate_clusters()
kmeans = kmeanspp(k=3, n_iter = 25)
labels = kmeans.fit(X_test)

colors = ['red', 'green', 'blue']  # match number of clusters

plt.scatter(X_test[:, 0], X_test[:, 1], c=[colors[i] for i in labels], alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c=colors, marker="*", s=300, edgecolor='k')
plt.title("KMeans Cluster Test")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()




          

        