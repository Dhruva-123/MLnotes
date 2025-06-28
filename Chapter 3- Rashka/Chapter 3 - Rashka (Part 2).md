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

**Decision Trees for Regression:**
