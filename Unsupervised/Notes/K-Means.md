
K- means is an unsupervised learning algorithm that attempts to group data or find 'clusters' in the data. Therefore, this is called a clustering algorithm. Here are the steps on how it works:

- Say we have data with n features and m samples. If we assume each sample as a point in geometry, then, each point is n-dimensional. This whole algorithm runs on finding the Euclidean distance between these points. So it is neat to imagine these points on a cartesian plane. 
- Then, we take 'K' random points or samples from the given data (K is a hyperparameter). Giving K to the model tells the model to look for K different groups. 
- Now that we have K random points, we will assume that these points are the actual mean points of these K groups and we will find the Euclidean distance from every other point to each of these points. We will check which of these K points is closest to the given point and then, we will assign the group name. Say, point A is closest to group 5 than 4 3 2 or 1. Then, point A is grouped as 5. We do this for the whole dataset. 
- Once we are done, we will calculate the variance between the groups and store the variance in an array.
- We will continue this process a lot of times because finding the least variance is not a simple task. Once we are done with these iterations, we will check the least variance setup in these and assume that to be the correct clustering. 

There are a few critical points to note here:

- How do we find 'K'? :- Finding K is done via variance. We will assume K = 2, do the means process and calculate variance. same for K = 3 4 5 ... m (where m is the number of training samples). There is no doubt that the variance keeps decreasing with the increase in the value of K. However, for the optimal K, the shift of variance from K to K+1 will be very low and the graph of variance starts to trail from K. We, therefore, find that K with differentiation. The plot we use to find K is called an "elbow plot".
- For multidimensions? :- In the multidimensional case, the same Euclidean method works because we can find the distance between 2 points in any number of dimensions from it.
That should be enough theory for a python implementation of our own.



There is another variation of this algorithm and that is called k-means ++. It does the exact same thing as k-means, but here, the initialization is different. Here the centroids aren't initialized as a random point. But here, only the first centroid is taken randomly. The rest are taken via a probabilistic approach. This leads to faster convergence rates. But generally, this is a bit more computationally heavy. This is in general far more sophisticated and I assume the golden standard in the industry. So we will also attempt to implement that. 


We noticed that both our k-means and k-means++ worked well with 200 iterations. But as the iterations slowed down, k-means++ started to take over the process. Eventually, at 25 iterations, k-means made wrong classifications while k-means++ didn't. So, for low iteration rates, k-means++ is the way to go. If you have high iterations or a casual theme, you can go with k-means.


