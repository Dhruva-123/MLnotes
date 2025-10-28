# Different Validation techniques
## K-Fold Cross-Validation:
### Code:
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

## Stratified K-Fold :
### Code:
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


## Grouped K-Fold:
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

### Code:
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

## TimeSeriesSplits:

Now, this one must always be used for TimeSeries based models. What this does is, it takes data in a series of time and over each iteration of the for loop we are about to establish, it trains the model on  past events and tests them on the current events. That is easy enough to explain, Here is an illustrated example on how it works:

### Code:

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
### Output:
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


# Checking How Well Our Model Is Doing (Classification):

We usually use the words "Accuracy", "Precision", "Recall" interchangably. But in the context of serious ML, they are all very very different.

But First, Inorder to really understand these, let me set some ground definitions:

TP = True Positive. This means, if the label is positive and you've correctly labeled it as positive.
FP = False Positive. This means, if the label is negative and you've wrongly labeled it as positive.
TN = True Negative. This means, if the label is negative and you've correctly labeled it as negative.
FN = False Negative. This means, if the label is positive and you've wrongly labeled it as negative.

## Accuracy:

Accuracy (A) = (TP + TN)/(TP + TN + FN + FP)

This tells you this;
"Out of all the predictions, how many were right"
This is usually used when the two classes are roughly equal in number. This is the most used, normally.

---

## Precision:

Precision (P) = TP/(TP + FP)

This tell you;
"Out of all the things that you though were positive, how many were correct"
This is usually used when there is a particular class that you are very focused on if it's being classified right or wrong. EX: In email spam, we care more about not classifying the good one as spam rather than restricting spam harshly.

---

## Recall:

Recall (R) = TP/ (TP + FN)

This tells you;
"Out of all the positive ones, how many did you classify correctly."

EX: In cancer predictions, it is far more costly for a human to have cancer and it be classified as not cancer. It is fine if the guy doesn't actually have cancer but it says they do. This is usually used in flagging based models.


---
---
## F1 Score:

F1 Score = 2*(P*R)/(P + R)

where P is Precision and R is Recall

This is used when we care equally about precision and recall. This is usually used in hardcore situations where being correct is everything. But the fatal flaw is, it doesn't tell you what the problem is. It could either be in Precision or in Recall.

---
---
---

# Graphs

## ROC Curve (Receiver Operationing Characteristic):

X-Axis = TPR (True Positive Rate) = Recall
Y-Axis = FPR (False Positive Rate) = FP/(FP + TN)

The curve is always goes downward as the training goes on.

The area under the ROC curve (AUC) is also a critical concept to bare in mind. This clearly tells you how well the model is at seperating two given classes. The more the area under the AUC curve is, the greater the confidence of the model is. That high confidence is also not great for the user and we will learn how in some while.

|Use-case|Metric to Prioritize|
|---|---|
|Email spam filter|Precision|
|Cancer detection|Recall|
|Fraud detection|F1 Score (or Precision, depending on context)|
|Ad click prediction|Log Loss|
|Model ranking (IR tasks, etc.)|AUC|
Note that Log Loss is a heavily penalizing loss. It will absolutely kill you if you are wrong.

However, In regression, there are a lot of different types of tests:

# Checking How Well Our Model Is Doing (Regression):

## $R^2$ error:

$R^2$ error = 1 - $SS_{res}$/$SS_{tot}$

Where;
$SS_{res}$ = (y - y_pred)^2 --> Model Error
$SS_{tot}$ = (i - y_mean)^2 --> Mean based variance in the model.

This error simply shows how much better our model is at assigning values than just mean values.

$R^2$ < 0 => our model is shittier than average value 
$R^2$ = 0 => Our model is barely equal to average value guess
$R^2$ > 0 => Our model is better than average guess

## MSE:

We know about MSE, so we are gonna ignore it here.

## MAE:

Instead of squaring the errors like MSE, you put it in a modulus and get a positive value.

## Huber Loss:

**Huber Loss** transitions:
- From **MSE behavior** for small errors
- To **MAE behavior** for large errors

This gives **robustness** and **smooth gradient flow**.


### 🔢 Mathematical Formula

Let error $(e = y -\hat{y} )$

$$
L_\delta(e) =
\begin{cases}
\frac{1}{2} e^2 & \text{if } |e| \leq \delta \\
\delta(|e| - \frac{1}{2} \delta) & \text{if } |e| > \delta
\end{cases}
$$

Where:
- \($\delta$) is a tunable threshold that controls the transition between MSE and MAE

### 📊 Behavior Summary

| Error Magnitude | Behavior | Loss Function Type |           |          |
| --------------- | -------- | ------------------ | --------- | -------- |
| \(              | e        | $\leq \delta$      | Quadratic | Like MSE |
| \(              | e        | $> \delta$         | Linear    | Like MAE |
### ⚙️ Interpretation

| Feature                  | Huber Loss |
| ------------------------ | ---------- |
| Robust to outliers       | ✅          |
| Smooth gradients         | ✅          |
| Differentiable           | ✅          |
| Tunable via \( \delta \) | ✅          |

### ✅ When to Use Huber Loss?

Use **Huber Loss** in regression tasks when:
- You want **robustness** to outliers
- You still need **smooth optimization**
- You want a **hybrid between MAE and MSE**

### ⚠️ Notes

- Tuning \( \delta \) is crucial:
    - Small \( \delta \) → behaves like MAE
    - Large \( \delta \) → behaves like MSE
- Default \( \delta = 1 \) is common but may not suit all datasets






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
    