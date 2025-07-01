Random Forest — Detailed Notes
🔹 What Is a Random Forest?A Random Forest is an ensemble of decision trees where:Each tree is trained on a random bootstrap sample of the data (sampling with replacement).At each split in each tree, a random subset of features is considered, not the full set.The final output is based on majority vote (for classification) or average (for regression) of all trees' predictions.
🔹 Why Use Random Forests?
Main Benefit: Reduces overfitting (variance) of decision trees, while maintaining low bias.Key idea: A single tree overfits; a forest of diverse trees generalizes better.Random Forests achieve this through:
Bootstrap aggregation (bagging):
promotes model diversity.Feature randomness: reduces correlation between trees.Ensemble voting: stabilizes the final prediction.
🔹 Key Components
1. Bootstrapping:
    Each tree is trained on a randomly sampled subset (with replacement) from the training data.Sample size = size of original dataset (n_samples)This creates diversity across trees.
2. Random Feature Subsets:
    At each split:Instead of considering all d features, consider only k = sqrt(d) (for classification).For regression, k = d/3 is often used.Increases decorrelation between trees.
3. Tree Independence:
    Each tree is fully grown and unpruned.Deeper trees overfit — but in RF, overfitting is mitigated by ensembling.Each tree operates independently.
4. Ensemble AggregationClassification: majority voting across trees.Regression: mean of predictions from all trees.
🔹 Random Forest vs Decision Tree:
Feature Decision Tree Random ForestOverfitting High Low (due to averaging)Interpretability High LowVariance High ReducedBias Low Slightly increasedTraining Time Fast Slower (multiple trees)Predictive Power Lower Higher (robust)
🔹 Hyperparameters to Tune:
Hyperparameter Descriptionn_trees Number of trees in the forestmax_depth Maximum depth of each treemin_samples_split Minimum samples required to split a nodemax_features Number of features to consider at each splitbootstrap Whether to use bootstrap samples or notoob_score Whether to use out-of-bag samples for validation
🔹 Out-of-Bag (OOB) Error:
Since each tree is trained on a bootstrap sample, about 1/3 of the data is left out.These "out-of-bag" samples can be used to estimate the generalization error.Acts like built-in cross-validation.
🔹 When to Use Random Forests:
Use Random Forests when:You want strong performance out-of-the-box.You need a model that handles both numerical and categorical data.You want robustness to noise, outliers, and overfitting.You’re willing to trade interpretability for accuracy.
🔹 Limitations:
Slower to train (especially with large n_trees).Less interpretable than a single tree.Can still overfit noisy datasets if not tuned.Not ideal for high-dimensional sparse data (e.g., text classification — use linear models instead).
🔹 Strategic Perspective:
Random Forest =✅ Low variance✅ High accuracy✅ Simple to use (minimal tuning)❌ Noisy for interpretability❌ Slower than single treesIt is a variance-reduction machine that builds decorrelated models and aggregates them. In ensemble learning terms:Bias ↑ a bitVariance ↓ a lotGeneralization ↑