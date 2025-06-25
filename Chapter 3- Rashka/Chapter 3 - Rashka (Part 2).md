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
        Here, $p(i|t)$ is a term that represents the proportion of the example that belongs to class $i$ at a particular node $t$ . This lies between 0-1 and sum of all these $p(i|t)$  = 1.
    2. Gini Impurity($I_G$):
        $I_G$ = $\sum$$p(i|t)$($1 - p(i|t)$) = 1 - $\sum$$p(i|t)^2$ 
        Note that entropy impurity and gini impurity are very very close to each other and usually give the same output.
    3. Classification Error($I_E$):
        $I_E(t)$ = 1 - max{$p(i|t)$} 
        Usually, we use the classification error impurity check for "pruning" the tree rather than fitting the tree model. 