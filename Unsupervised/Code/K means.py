import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k = 3): ## Here K is the number of groups that need to be found in the data.
        self.k = k
        self.centroids = None ## This is where we store the centres of all the groups that we found.
    
    def euclidean_distance(self, point, random_centers): ## This is simple Euclidean distance.
        return np.sqrt(np.sum((random_centers - point)**2, axis = 1))
    

    def fit(self, X, n_iter):
        if not isinstance(X, np.ndarray): ## The input can be both numpy array or a normal python list. 
            X = np.array(X)
       
        self.centroids = np.random.uniform(np.amin(X, axis = 0), np.amax(X, axis = 0), size = (self.k, X.shape[1])) ###Initial setup of centers. We are asssigning random centres such that they are in the range of possibility.
        
        for _ in range(n_iter):
            y = []
            for point in X:
                distances = self.euclidean_distance(point, self.centroids)
                label = np.argmin(distances)
                y.append(label)
            y = np.array(y) ## This is the array we are returning. Here, we store the particular group index that we are assiging to each row of the given data. 
            cluster_centers = []

            for i in range(self.k): ## For each group, we are attempting to find out the indicies inside that particular group and then do operations on that particular group.
                    cluster_centers.append(np.argwhere(y == i)) ## creating groups in cluster)centers            
            calculated_centers = []
            for i, points in enumerate(cluster_centers): ## Here 'i' stands for the group index and points stands for rows in the group. 
                 if len(points) == 0: ## if there are no elements in a group, we tend to give it the previous setup.
                      calculated_centers.append(self.centroids[i])
                 else:
                      calculated_centers.append(np.mean(X[points.flatten()], axis = 0)) ## np.mean usually gives the wrong sort of dimensions, so we need to flatten it for our use. 
            calculated_centers = np.array(calculated_centers)
            if np.max(np.abs(self.centroids - calculated_centers)) >= 0.001: ### abs because error could be both positive or negative and max because even if one part is off by a lot, we fix it.
                self.centroids = calculated_centers
        return y

### This is just to generate clusters.
def generate_clusters(n_points_per_cluster=50, centers=None, std=5):
    """
    Generate synthetic data for testing KMeans
    """
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
kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X_test, n_iter=25)

colors = ['red', 'green', 'blue']  # match number of clusters

plt.scatter(X_test[:, 0], X_test[:, 1], c=[colors[i] for i in labels], alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c=colors, marker="*", s=300, edgecolor='k')
plt.title("KMeans Cluster Test")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
