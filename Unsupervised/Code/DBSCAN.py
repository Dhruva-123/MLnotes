import numpy as np

class DBSCAN:
    def __init__(self, eps = 0.5, min_samples = 10, metric = "Euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels = None

    def distance_calculator(self, a, b):
        if self.metric == "Euclidean":
            return np.sqrt(np.sum((a - b)**2, axis = 1))
        elif self.metric == "Manhattan":
            return np.sum(np.abs((a - b)), axis = 1)
        else:
            raise ValueError("metric unknown : {self.metric}. Change metric to either 'Euclidean' or 'Manhattan'")

    def neighbor_finder(self, X, point):
        distances = self.distance_calculator(X, point)
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors

    def fit(self, X):
        if isinstance(X, np.ndarray):
            X = np.array(X)
        no_samples = X.shape[0]
        self.labels = np.full(no_samples, -1)
        cluster_id = 0
        visited = np.zeros(no_samples, dtype = bool)
        for i in range(no_samples):
            if visited[i] == True:
                continue
            visited[i] = True
            neighbors = self.neighbor_finder(X, X[i])
            if len(neighbors) < self.min_samples:
                continue
            self.labels[i] = cluster_id
            seeds = list(neighbors.copy())
            seeds.remove(i)
            while seeds:
                current = seeds.pop()
                if visited[current] == False:
                    visited[current] = True
                    new_neighbors = self.neighbor_finder(X, X[current])
                    if len(new_neighbors) >= self.min_samples:
                        seeds.extend([n for n in new_neighbors if n not in seeds])
                if self.labels[current] == -1:
                    self.labels[current] = cluster_id
            cluster_id += 1
        return self.labels
    
# --- Testing DBSCAN on synthetic data ---
def generate_clusters(n_points_per_cluster=50, centers=None, std=1.0):
    if centers is None:
        centers = [[0,0],[5,5],[10,0]]
    X = []
    for c in centers:
        X.append(np.random.randn(n_points_per_cluster,2) * std + c)
    return np.vstack(X)

# Example usage
X_test = generate_clusters()
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit(X_test)

print("Cluster assignments:")
print(labels)



        