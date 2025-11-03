hierarchical clustering is usually used for heatmaps. 

There are two types of hierarchical clustering:

- Agglomerative (This is the most common one) starts with each point as it's own cluster, and then builds a tree up from there. They merge the closest one's in each step of the tree and when we want to derive a particular number of clusters, we can prune the tree upto that part and get the clusters we need. 
- Divisive (must more niche) This starts with every element under one cluster and then we break the clusters into groups till every point is its own cluster. Once this is done, we will prune the tree up to the point where we get the number of clusters we desire or the a desired depth of the tree. 

First, we will talk about Agglomerative because that is the most common. Here is a step by step show case of how it goes:

First, we start off with each point as it's own cluster. After that, we measure pair wise distances between these clusters. Then, we merge the closest clusters. We then repeat this process until every element is in a single cluster.


In theory, this is pretty simple. But practically, we need to know a few more things. 

#### Linkage Methods

Say we are trying to calculate the distance between clusters. Linkage talks about the method we use to calculate that distance. There are several different methods we use to calculate that distance. Here is a table of these methods, each of them have their own trade offs:

|Linkage Type|Distance Formula|Intuition|
|---|---|---|
|**Single Linkage**|min(distance between any two points in clusters)|Tends to form long “chains”|
|**Complete Linkage**|max(distance between any two points)|Creates compact, tight clusters|
|**Average Linkage**|mean(distance between all pairs)|Balance between single and complete|
|**Centroid Linkage**|distance between cluster centroids|Can cause cluster inversion (not monotonic)|
|**Ward’s Method**|minimizes total variance increase|Often best for numeric data|

Note that this inverted sort of merging tree is called a "Dendrogram". 
Also note that this 'Linkage' is different from distance calculation metrics. For distance calculation metrics, we have Euclidean distance, Manhattan distance, Cosine distance etc etc. This is the formula you use to calculate the distance. 

Linkage is on what basis we merge these points, is it the avg distance between them,  of the highest distance or something else.


So, usually for this particular type of clustering technique, we have to take a lot of parameters from the user. Here is a sample of the parameters we need to take in:

```
HierarchicalClustering(
    linkage="average",
    metric="euclidean",
    distance_threshold=None,
    n_clusters=3
)
```

Note that the `distance_threshold` is a parameter which takes in a number and sets that as a threshold. It stops the model from merging clusters if they have that distance or more between them with the linkage and metric we are using. 
