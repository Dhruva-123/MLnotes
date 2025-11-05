## 1. Overview

DBSCAN is a **density-based clustering algorithm** that identifies clusters based on the density of points. It can discover clusters of **arbitrary shape** and can identify **noise points** (outliers).

**Key concepts**:

- **Core Point:** A point with at least `min_samples` neighbors within distance `eps`.
    
- **Border Point:** A point within `eps` of a core point but has fewer than `min_samples` neighbors.
    
- **Noise Point:** A point that is neither a core point nor a border point.
    

---

## 2. Parameters

|Parameter|Description|
|---|---|
|`eps`|Maximum distance between two points to be considered neighbors.|
|`min_samples`|Minimum number of points required to form a dense region (core point).|
|`metric`|Distance metric used (commonly Euclidean).|

**Intuition:**

- Small `eps` → fragmented clusters; many points may be marked as noise.
    
- Large `eps` → clusters merge; risk of one giant cluster.
    

---

## 3. Algorithm Steps

1. Start with all points unvisited.
    
2. Pick an unvisited point `p`:
    
    - Find all points within distance `eps` → `neighbors`.
        
    - If `len(neighbors) < min_samples`, mark as **noise**.
        
    - Else, start a new cluster.
        
3. Expand the cluster:
    
    - For each point in the cluster’s `neighbors`:
        
        - If unvisited, mark as visited and find its neighbors.
            
        - If it has enough neighbors (`>= min_samples`), add them to the cluster.
            
        - Assign cluster label if previously noise.
            
4. Repeat until all points are visited.
    

---

## 4. Distance Calculations

For a point `pip_ipi​` and dataset `XXX`:

`distance(pi,Xj)=∑k=1d(Xj,k−pi,k)2\text{distance}(p_i, X_j) = \sqrt{\sum_{k=1}^d (X_{j,k} - p_{i,k})^2}distance(pi​,Xj​)=k=1∑d​(Xj,k​−pi,k​)2​`

- For Euclidean distance (`metric='euclidean'`), or
    
- Manhattan distance (`metric='manhattan'`) can also be used.
    

---

## 5. Cluster Formation Table

|Step|Description|
|---|---|
|Identify neighbors|Points within `eps` of a core point.|
|Check density|Count neighbors to determine core vs. border.|
|Expand cluster|Add neighbors recursively if they are core points.|
|Label noise|Points not in any cluster remain labeled as `-1`.|


## 7. Notes on Parameter Selection

- `eps` controls cluster granularity.
    
- `min_samples` defines the minimum density for a core point.
    
- Use a **k-distance graph** to select `eps`:
    
    1. Compute distance to the `k-th` nearest neighbor for each point (`k = min_samples - 1`).
        
    2. Sort distances and plot them.
        
    3. Pick `eps` at the **elbow** point where distances start increasing sharply.
        

---

## 8. Output

- Points are labeled with **cluster indices**: 0, 1, 2, …
    
- Noise points are labeled `-1`.
    
- Clusters may have **arbitrary shape**, unlike k-means.