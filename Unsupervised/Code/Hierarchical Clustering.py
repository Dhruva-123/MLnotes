import numpy as np
from itertools import combinations

class HierarchicalClustering:
    def __init__(self, metric = "Euclidean", linkage = "ward", n_clusters = None, threshold = None):
        self.metric = metric
        self.linkage = linkage
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.labels = None

    def pairwise_dist(self, cluster_a, cluster_b):
        if self.metric == "Euclidean":
            diff = cluster_a[:,None,:] - cluster_b[None, :, :]
            return np.sqrt(np.sum(diff**2, axis = 2))
        elif self.metric == "Manhattan":
            diff = cluster_a[:,None,:] - cluster_b[None, :, :]
            return np.sum(np.abs(diff), axis = 2)
        else:
            raise ValueError("Unknown Metric: {self.metric}. Cannot proceed.")
        
    def cluster_dist(self, X, indicies_a, indicies_b):
        cluster_a = X[np.array(indicies_a)]
        cluster_b = X[np.array(indicies_b)]
        if self.linkage == "single":
            dist = self.pairwise_dist(cluster_a, cluster_b)
            return float(np.min(dist))
        elif self.linkage == "complete":
            dist = self.pairwise_dist(cluster_a, cluster_b)
            return float(np.max(dist))
        elif self.linkage == "average":
            dist = self.pairwise_dist(cluster_a, cluster_b)
            return float(np.mean(dist))
        elif self.linkage == "centroid":
            center_a = np.mean(cluster_a, axis = 0)
            center_b = np.mean(cluster_b, axis = 0)
            return float(np.linalg.norm(center_a - center_b))
        elif self.linkage == "ward":
            center_a = np.mean(cluster_a, axis = 0)
            center_b = np.mean(cluster_b, axis = 0)
            len_a = len(cluster_a)
            len_b = len(cluster_b)
            return float(((len_a * len_b)/(len_a + len_b))*np.sum((center_a - center_b)**2))
        else:
            raise ValueError("Unknown Linkage: {self.linkage}. Cannot Proceed")
    
    def fit(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        clusters = {i : [i] for i in range(len(X))}
        next_cluster_index = len(X)
        merged = set()
        while True:
            if self.n_clusters is not None and self.n_clusters >= (len(clusters)):
                break 
            ids = list(clusters.keys())
            min_dist = np.inf
            cluster_pairs = None
            for i,j in combinations(ids, 2):
                dist = self.cluster_dist(X, clusters[i], clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    cluster_pairs = (i , j)
            
            if not cluster_pairs:
                break
            
            if self.threshold is not None and self.threshold < min_dist:
                break

            i, j = cluster_pairs

            merged_cluster = clusters[i] + clusters[j]
            clusters[next_cluster_index] = merged_cluster
            next_cluster_index += 1
            merged.add(i)
            merged.add(j)
            clusters = {cid: cluster for cid, cluster in clusters.items() if cid not in merged}
        
        cluster_ids = list(clusters.keys())
        labels = np.empty(len(X), dtype = int)
        for label, id in enumerate(cluster_ids):
            labels[clusters[id]] = label
        self.labels = labels
        return labels
    

def generate_clusters(n_points_per_cluster=5, centers=None, std=1):
    if centers is None:
        centers = [[0,0],[5,5],[10,0]]
    X = []
    for c in centers:
        X.append(np.random.randn(n_points_per_cluster,2)*std + c)
    return np.vstack(X)

X_test = generate_clusters()
hc = HierarchicalClustering(n_clusters=3)
labels = hc.fit(X_test)

print("Cluster assignments:")
print(labels)

