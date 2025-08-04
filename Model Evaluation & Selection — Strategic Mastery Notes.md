K-Fold Cross-Validation:
    Code:
    This is how you import KFold
         
         from sklearn.model_selection import KFold
    This is how you use it
        X, y = load_iris(return_X_y=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
         y_pred = model.predict(X_val)

Stratified K-Fold :
     Code:
         This is how you import it
         
         from sklearn.model_selection import StratifiedKFold

         This is how you use it

          X, y = load_iris(return_X_y=True)
          skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
          for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)


Grouped K-Fold:
Say we use normal KFold in a dataset like this;
|Sample ID|User ID|Features (`X`)|Clicked (`y`)|
|---|---|---|---|
|1|U1|[..]|1|
|2|U1|[..]|0|
|3|U2|[..]|1|
|4|U2|[..]|1|
|5|U3|[..]|0|
|6|U3|[..]|1|

And our KFold splits the data like this, train = 1,3,5; test = 2,4,6
in this, the major problem is, if we train a model on this data and use the test for test, The model clearly learns all about User 1 and User 2 and User 3 before hand, so the results will be very easily over stated. But in groupKFold, what ends up happening is, the computer will split the data based on groups of User ID and then we train and test. This new train and test looks like this;
train = 1,2,3,4 ; test = 5,6. The computer hasn't seen group 3 yet, now if we test it on that, itll be more valid.

Code:
     Here is how we import it:
    
    from sklearn.model_selection import GroupKFold
    
	 Here is how we use it:
         
    X = df[['feature1']].values
    y = df['target'].values
    groups = df['subject_id'].values
    gkf = GroupKFold(n_splits=3)
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        print("Train groups:", df.iloc[train_idx]['subject_id'].unique())
        print("Test groups:",  df.iloc[test_idx]['subject_id'].unique()) 

TimeSeriesSplits:

Now, this one must always be used for TimeSeries based models. What this does is, it takes data in a series of time and over each iteration of the for loop we are about to establish, it trains the model on  past events and tests them on the current events. That is easy enough to explain, Here is an illustrated example on how it works:

from sklearn.model_selection import TimeSeriesSplit
import numpy as np

X = np.arange(10).reshape(-1, 1)   # features: [0], [1], ..., [9]
y = np.arange(10)                 # targets:   0, 1, ..., 9

tscv = TimeSeriesSplit(n_splits=3)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold + 1}")
    print("Train indices:", train_idx)
    print("Val indices:  ", val_idx)
    print("---")


Output:


Fold 1
Train indices: [0 1 2 3]
Val indices:   [4 5]

Fold 2
Train indices: [0 1 2 3 4 5]
Val indices:   [6 7]

Fold 3
Train indices: [0 1 2 3 4 5 6 7]
Val indices:   [8 9]


If you look at the example carefully, youll understand it very well.
also, a few points to note:
The n_splits here means the number of times we are running the loop. not the ratio in which we are splitting train test. The test example number is fixed here;
test_size = total_no_of_samples/(n_splits + 1)







# Quick Revision
## 🔁 1. Cross-Validation (CV) Schemes

### 📌 **Goal**: Estimate generalization performance reliably under various data assumptions.

### 🔹 **a. k-Fold Cross-Validation**

- **Process**: Split dataset into _k_ equal folds → train on _k−1_ folds → validate on remaining fold → average results.
    
- **Pros**: Low variance estimate with moderate _k_ (e.g., 5 or 10).
    
- **Cons**: Can leak label distribution if imbalanced data.
    
- **When to use**: General-purpose; no group/temporal structure.
    

### 🔹 **b. Stratified k-Fold**

- **Process**: Like k-Fold, but ensures **each fold preserves class proportions**.
    
- **When**: Classification tasks with imbalanced classes.
    
- **Key API**: `StratifiedKFold` in scikit-learn.
    

### 🔹 **c. Group k-Fold**

- **Process**: All data points from a single group (e.g., user/session/patient) appear in only one fold.
    
- **Prevents**: Data leakage from correlated samples.
    
- **Key API**: `GroupKFold` in scikit-learn.
    

### 🔹 **d. Time Series Split (Rolling Window)**

- **Process**: Folds respect time order (no future leakage).
    
- **Options**: Fixed or expanding training window.
    
- **Key use**: Forecasting, online learning, causal modeling.
    
- **Key API**: `TimeSeriesSplit`
    

---

## ⚖️ 2. Bias–Variance Decomposition

### 📌 **Core Equation**

Expected Error=Bias2+Variance+Irreducible Noise\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}Expected Error=Bias2+Variance+Irreducible Noise

### 🔹 **Bias**

- Model too simple (e.g., linear for quadratic data).
    
- Underfits training data.
    
- Inflexible hypothesis class.
    

### 🔹 **Variance**

- Model too complex (e.g., deep tree).
    
- Overfits training data noise.
    
- High sensitivity to data fluctuations.
    

### 🔹 **Use-cases**

- Diagnose learning behavior.
    
- Explain poor performance even when CV error is low.
    

---

## 📈 3. Learning Curves

### 📌 **Definition**

- **Plot**: Training and validation error vs. training set size.
    
- **Shape interpretation**:
    
    - **High bias**: both curves high and close.
        
    - **High variance**: large gap between training and validation curves.
        
    - **Good fit**: both low, minimal gap.
        

### 🔹 **Strategic uses**

- Estimate benefit from adding more data.
    
- Decide whether to regularize or increase model complexity.
    
- Helps prevent premature optimization.
    

---

## 🎯 4. Task-Specific Metrics

---

### 🟦 **A. Classification Metrics**

|Metric|Formula|Use-case|
|---|---|---|
|**Accuracy**|(TP+TN)/(TP+FP+TN+FN)|Only for balanced classes.|
|**Precision**|TP / (TP + FP)|Minimize false positives (e.g., spam filter).|
|**Recall**|TP / (TP + FN)|Minimize false negatives (e.g., cancer diagnosis).|
|**F1 Score**|2 × (Prec × Rec) / (Prec + Rec)|Balanced measure when classes are imbalanced.|
|**ROC Curve**|TPR vs. FPR|Threshold-independent metric.|
|**AUC**|Area under ROC|Summary metric for separability.|
|**Log Loss (Cross-Entropy)**|−∑yilog⁡pi-\sum y_i \log p_i−∑yi​logpi​|Penalizes incorrect confident predictions. Great for probabilistic models.|

#### 🔹 Interpretations:

- **Precision > Recall** → Conservative predictions.
    
- **Recall > Precision** → Aggressive detection.
    
- **Log loss vs. AUC** → Log loss for calibration, AUC for ranking.
    

---

### 🟩 **B. Regression Metrics**

|Metric|Formula|Use-case|
|---|---|---|
|**MSE**|1n∑(y−y^)2\frac{1}{n} \sum (y - \hat{y})^2n1​∑(y−y^​)2|Sensitive to outliers, good for training.|
|**RMSE**|MSE\sqrt{\text{MSE}}MSE​|Same unit as output.|
|**MAE**|(\frac{1}{n} \sum|y - \hat{y}|
|**Huber Loss**|Quadratic when small error, linear when large|Hybrid of MSE + MAE.|
|**R²**|1−SSresSStot1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}1−SStot​SSres​​|% variance explained. Not always meaningful in nonlinear or non-centered data.|

#### 🔹 Strategic insight:

- **MSE** is smooth and differentiable — great for gradient-based training.
    
- **MAE** is non-differentiable at 0 — harder to optimize.
    
- **Use Huber loss** in noisy regression settings.
    

---

### 🟨 **C. Ranking Metrics**

#### 🔹 **MAP (Mean Average Precision)**

- Measures average precision across queries/documents where **order of relevant items matters**.
    
- Higher = better top-level relevance across multiple lists.
    

#### 🔹 **NDCG (Normalized Discounted Cumulative Gain)**

DCGp=∑i=1prelilog⁡2(i+1)NDCGp=DCGpIDCGp\text{DCG}_p = \sum_{i=1}^p \frac{rel_i}{\log_2(i+1)} \quad \text{NDCG}_p = \frac{DCG_p}{IDCG_p}DCGp​=i=1∑p​log2​(i+1)reli​​NDCGp​=IDCGp​DCGp​​

- **Gain**: More relevant items near top → higher score.
    
- Normalized → scores between 0 and 1.
    

#### 🔹 Use-case

- Search ranking, recommender systems, feeds.
    
- Required where **relative order** matters more than just correct prediction.