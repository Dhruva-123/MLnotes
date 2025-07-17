**Decision Trees:**

Decision Trees are Tree like structres that are split into several branches and at each node, we make a decision to classifiy given example into a class. there are a few fundamental ideas in decision trees. 

1. Information Gain:
    Information Gain is the idea of gaining purity with each tree split. Purity is the idea 
    Information Gain (IG) = $I$($D_p$) - $\sum$ $I$($D_i$) * $n_i$/$n_p$
    Here, $I$ is called impurity function. Mathematically, Information Gain is the difference between the impurity of the parent node and the sum of fractions of impurities of each child node.
2. Impurity:
    There are 3 main types of impurity functions that are used to measure the impurity at a particular node. 
    1. Entropy ($I_H$):
        $I_H$ = - $\sum$ $p(i|t)$$log_2(p(i|t))$ 
        Here, $p(i|t)$ is a term that represents the proportion of the example that belongs to class $i$ at a particular node $t$ . This lies between 0-1 and sum of all these $p(i|t)$  = 1. Max entropy = 1 and least entropy is 0.
    2. Gini Impurity($I_G$):
        $I_G$ = $\sum$$p(i|t)$($1 - p(i|t)$) = 1 - $\sum$$p(i|t)^2$ 
        Note that entropy impurity and gini impurity are very very close to each other and usually give the same output.
    3. Classification Error($I_E$):
        $I_E(t)$ = 1 - max{$p(i|t)$} 
        Usually, we use the classification error impurity check for "pruning" the tree rather than fitting the tree model. 

Here is the fundamental Concept behind a Decision Tree:

1. A Decision Tree contains two types of nodes: 1. Decision Node 2. Leaf Node
2. Decision Node is where we make decision on weather to split the data or not, how to split it and the information gain calculation. The leaf node contains the value of the class that is higher in number. When we arrive at a leaf, we assign that leaf value to the data point and hence, we classify.
3. If a node contains points of a particular class and only that class, that node is called a pure node. Usually, leaf nodes are pure nodes.
4. We make the computer learn the best splitting conditions based on the information gain maximization. That is why this is called a machine learning algorithm and an if else tree.
5. So basically, what we are doing is, we take every feature in the given dataset and then we split based on that feature. Now, we check the feature value of every example of a particular feature and use it as threshold, we then calculate the IG and use the one that has max IG.

This model has two different classes:
1. node
2. tree


A few more points to note:
1. For keeping things in the finite split zone, we define a variable called the max_depth. Our intention is to make sure that no matter what, that tree cannot go bigger than the max depth at any branch. That is our idea. Max_depth is a hyperparameter that we tweak for efficiency's sake. 
2. We also maintain a min_sample variable. This makes sure that we don't split a given node further even if the sample size is less than a particular value. This is edge case management.
3. This might be the most important point here, If information gain is zero, then,  we are splitting a node that is already pure. We don't want that.



# Decision Trees 

- Random Forests built with a good Descision Tree can fit complex datasets easily, but, ofc, over fitting is a major risk
- We use regularization to stop over fitting. This can include a lot of different things:
- The Algorithm of Descision Trees are is called CART (Classification and Regression Trees). It's that recursive splitting algorithm.
- In tree splitting, if the min_samples is too low, the tree won't be regularized properly and that could lead to serious overfitting. It starts to pick up on the noise in the data and not the useful matter. So, with min_samples, bias increases and variance decreases.
- Random Forests improve the overall performance of a Decision Tree style because, They tend to reduce overfitting and hence variance.
## Limitations:

- Apperently, finding the most optmial tree configurations for a given dataset is an NP-complete problem. IDK what an NP-complete problem is. It is something along the lines of "solving it is hard, but checking the answer is easy". This is one of the limitations of a decision tree.
- Trees suck at rotation based structures. Say we have a dataset and a tree sucks at classifying it. Then, when we rotate the dataset, suddenly, the tree is perfect. This sort of dependancy of the orientation of the data is a key weakness in trees.

## Good to know:

- Trees are almost never powerful on their own. In the modern day, they are used with random forests or some sort of boosting procedures to get real value. But, they are indeed one of the most powerful tools when paired with these things. These are used in competitions regularly.
- Decision Trees are one of the most simple things to use because they don't need any data prep at all. Like None. No normalization of data is required and no feature scaling is required.
- Models like Decision Trees are easy to visualize and understand and see exactly what's going on inside. These are called white-box-models. In some cases, like neural nets, etc, we cannot really see what's going on, they are called black-box-models. In white box ones, we can clearly see what made the model decide on a particular class or regression but in black box ones, its not soo easy.
- We can also bring out the probabilites of these classes occurances by simply finding out the ratio of a particular class in the leaf node we ended up in. It's quite simple, really.
- The time and space complexity of a decision tree when in use is very optimal. In fact, it is just O(log(m)) where m is the number of features. But the trouble comes when we are building a decision tree. Here, we will have to loop over every feature, that takes n*m*log(m). therefore, we usually sort the dataset first and then threshold it. but still, itll be n*log(m)*log(n). So yes. it is what it is.
- Gini impurity is usually a little faster than entropy one, but entropy one is usually a little more balanced than Gini one. so, either way, it doesn't matter.
- Models like decision trees are called non-parametric models. It doesn't mean they dont have any parameters, it only means that they make new parameters during run time (split pararmeters). The problem with this type of modeling is, they tend to overfit to the training data. in order to stop that overfitting, we use regularization hyper parameters, like max_depth and such. These do stop the overfitting, ofc, but not soo much. There are other algorithms that train a tree without any bounds and then prune the tree to get useful trees.
- A node whose children are all leaf nodes is considered unnecessary if the purity improvement it provides is not statistically significant. Standard statistical tests, such as the χ2 test, are used to estimate the probability that the improvement is purely the result of chance (which is called the null hypothesis). If this probability, called the pvalue, is higher than a given threshold (typically 5%, controlled by a hyperparameter), then the node is considered unnecessary and its children are deleted. The pruning continues until all unnecessary nodes have been pruned.
