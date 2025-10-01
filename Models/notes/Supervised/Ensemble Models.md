- A dataset is also called a sample.
- The learned model is called a hypothesis
- Tree pruning is of 2 types: pre-pruing and post-pruing. More on that later.
- Ensemble methods learn multiple algorithms to attain the end product. Not relying on a single model

# AdaBoost: Deep Theory Notes  
**Author:** Rahul Dhruva  

---

## Philosophy of AdaBoost

AdaBoost (Adaptive Boosting) is an ensemble technique that converts weak learners into a strong learner by focusing on misclassified instances over iterations. Originally proposed by Freund & Schapire (1996).

---

## Problem Setup

Binary classification dataset:  
$$(x_1, y_1), ..., (x_m, y_m), \quad y_i \in \{-1, +1\}$$

Final classifier:
$$
H(x) = \text{sign}\left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)
$$

---

## AdaBoost.M1 Algorithm

1. **Initialize weights**:
   $$
   D_1(i) = \frac{1}{m}
   $$

2. **For $t = 1$ to $T$**:
   - Train weak classifier $h_t(x)$ using $D_t$
   - Compute error:
     $$
     \varepsilon_t = \sum_{i=1}^{m} D_t(i) \cdot \mathbb{I}[h_t(x_i) \ne y_i]
     $$
   - Compute model weight:
     $$
     \alpha_t = \frac{1}{2} \log \left( \frac{1 - \varepsilon_t}{\varepsilon_t} \right)
     $$
   - Update weights:
     $$
     D_{t+1}(i) = \frac{D_t(i) \cdot e^{-\alpha_t y_i h_t(x_i)}}{Z_t}
     $$
     where:
     $$
     Z_t = \sum_{i=1}^m D_t(i) \cdot e^{-\alpha_t y_i h_t(x_i)}
     $$

3. **Final prediction**:
   $$
   H(x) = \text{sign}\left( \sum_{t=1}^{T} \alpha_t h_t(x) \right)
   $$

---

## Loss Minimization View

AdaBoost minimizes **exponential loss**:
$$
\mathcal{L}_{\text{exp}}(F(x)) = \sum_{i=1}^m e^{-y_i F(x_i)}
$$

Where:
$$
F(x) = \sum_{t=1}^T \alpha_t h_t(x)
$$

---

### Why Exponential Loss?

- Strongly penalizes confident misclassifications  
- Theoretically elegant  
- Computationally convenient  

---

### Gradient Boosting Connection

- AdaBoost = gradient boosting with exponential loss  
- Each iteration = function space gradient descent

---

## Understanding $\alpha_t$

$$
\alpha_t = \frac{1}{2} \log\left( \frac{1 - \varepsilon_t}{\varepsilon_t} \right)
$$

- Higher $\alpha_t$ → more trust in $h_t$
- If $\varepsilon_t > 0.5$, flip prediction:
  $$
  h_t(x) \leftarrow -h_t(x)
  $$

---

## Margin Theory

Margin:
$$
\text{margin}(x_i) = \frac{y_i F(x_i)}{\sum_{t=1}^T \alpha_t}
$$

- Higher margin → better generalization
- AdaBoost increases margin distribution over time

---

## Theoretical Guarantees

### Training Error:
$$
\text{Error}_{\text{train}} \leq \prod_{t=1}^{T} Z_t
$$

- $Z_t < 1$ leads to exponential decay

### Generalization:
- Controlled via margin theory  
- AdaBoost maintains high generalization even after overfitting training data

---

## Practical Enhancements (from your work)

- **Flipping**: When $\varepsilon_t > 0.5$, flipping $h_t(x)$ increases ensemble performance  
- **Noise**: Your tree outperforms sklearn’s in high noise; AdaBoost helps both  
- **Robustness**: AdaBoost improves margin separation and reduces test error even in noisy settings

---

## Variants of AdaBoost

- **AdaBoost.M1**: Multiclass
- **Real AdaBoost**: Confidence-rated predictions
- **LogitBoost**: Uses logistic loss
- **GentleBoost**: Better noise tolerance
- **AdaBoost.MH**: Multi-label

---

## Your Tree vs Scikit-Learn

### Your Tree:
- Random top-k thresholding
- Gini/Entropy
- Better regularization → better under noise

### Scikit-learn Tree:
- Optimized splits
- Overfits when shallow
- Needs careful tuning

**Your finding**:  
- Custom tree > sklearn tree under noise  
- AdaBoost + sklearn tree > both

---

## Limitations of AdaBoost

- Sensitive to noisy labels  
- Weak learners must be better than chance  
- High training cost (sequential)  
- No built-in regularization  

## 🔍 AdaBoost Deep Addendum: What You Missed (But Need)

These are essential advanced concepts not yet in your notes — meant to close every remaining gap in your AdaBoost understanding.

---

### 🧠 Geometric Interpretation

AdaBoost incrementally builds a decision boundary by:
- Placing weak cuts in regions where the current ensemble fails
- Gradually shaping the final decision surface by stacking corrections

> Visualize it as “carving out the error” with sequential hyperplanes

---

### 📉 Bias-Variance Behavior

AdaBoost shows unusual dynamics:

- **Bias** drops quickly
- **Variance** stays relatively stable, only rising mildly in later stages

This contrasts with typical high-capacity models:
- Boosting doesn’t overfit immediately
- Training error → 0  
- Test error keeps improving for many rounds

> Boosting can *simultaneously* reduce bias and variance — rare but empirically observed

---

### ⛔ Regularization and Early Stopping

AdaBoost has **no built-in regularization**  
This means:
- It keeps fitting even on noise
- Can overfit when weak learners are too expressive

Mitigation strategies:
- **Early stopping**
- **Alpha clipping**:
  
 $\alpha_t = \min(\alpha_t, \alpha_{\text{max}})$



---


### 🔁 Functional Coordinate Descent View

AdaBoost is equivalent to coordinate descent in function space:

- Each base learner $h_t(x)$ is a direction
    
- $\alpha_t h_t(x)$ is a step along that direction
    

This connects AdaBoost to optimization theory:

> “Boosting as gradient descent in function space with exponential loss”

---

### 🔒 Logistic Regression Connection

While AdaBoost minimizes exponential loss:

$Lexp=∑iexp⁡(−yiF(xi))\mathcal{L}_{\text{exp}} = \sum_i \exp(-y_i F(x_i)) Lexp​=i∑​exp(−yi​F(xi​))$

LogitBoost minimizes:

$$Llog=∑ilog⁡(1+exp⁡(−yiF(xi)))\mathcal{L}_{\text{log}} = \sum_i \log(1 + \exp(-y_i F(x_i))) Llog​=i∑​log(1+exp(−yi​F(xi​)))$$

LogitBoost:

- Is more noise-robust
    
- Has probabilistic interpretation
    
- Improves stability when confidence matters
    

---

### 🧪 Multi-Class Extensions

#### SAMME (Discrete Classifier)

- Boosts multiclass classifiers directly
    
- Requires weak learners to beat random chance
    

#### SAMME.R (Real-Valued Confidence)

- Uses probability/confidence estimates
    
- Lower variance
    
- Generalizes Real AdaBoost to multiclass
    

---

### ⚠️ Numerical Stability Trick

To avoid infinite $\alpha_t$:

$$εt=clip(εt,1e−10,1−1e−10)\varepsilon_t = \text{clip}(\varepsilon_t, 1e^{-10}, 1 - 1e^{-10}) εt​=clip(εt​,1e−10,1−1e−10)$$

You already implemented this — ✅ excellent move.

---

### 🧱 Additive Model View

AdaBoost builds:

$$F(x)=∑tαtht(x)F(x) = \sum_t \alpha_t h_t(x) F(x)=t∑​αt​ht​(x)$$

This is a **forward stagewise additive model** (FSAM)

At each round:

$$Ft+1(x)=Ft(x)+αtht(x)F_{t+1}(x) = F_t(x) + \alpha_t h_t(x) Ft+1​(x)=Ft​(x)+αt​ht​(x)$$

Think of this as building a function like stacking Lego bricks — each block targets the last block’s mistakes.

---

### 🧪 Margin Tracking

To monitor classifier confidence per sample:

$$margin(xi)=yi⋅F(xi)∑tαt\text{margin}(x_i) = \frac{y_i \cdot F(x_i)}{\sum_t \alpha_t} margin(xi​)=∑t​αt​yi​⋅F(xi​)​$$

Plotting margin histograms over time:

- Helps debug underfitting vs overfitting
    
- Explains generalization power even after zero training error

1.AdaBoost can be interpreted as maximizing the **minimum margin** under a normalization   constraint on $\alpha_t$, a dual to SVM-style formulations.
2.$Z_t$ connects to probabilistic interpretations — log-loss and exponential loss both emerge from exponential families. In LogitBoost, this becomes explicit.
3.If base learners are too deep, they can memorize noise, and boosting amplifies these patterns.
- You're light on **Real AdaBoost**, which builds:

$$ht(x)∈[−1,1]$$,as confidence-rated $$predictionh_t(x) \in [-1, 1], \quad \text{as confidence-rated prediction}ht​(x)∈[−1,1],as confidence-rated prediction$$

And updates with:

$$αt=12log⁡(1+ht(x)1−ht(x))\alpha_t = \frac{1}{2} \log \left( \frac{1 + h_t(x)}{1 - h_t(x)} \right)αt​=21​log(1−ht​(x)1+ht​(x)​)$$

📌 **Add:**

> Real AdaBoost smooths the margin distribution and has a closer link to log-likelihoods — connects better with logistic regression and Bayesian models.


AdaBoost implicitly promotes diversity — because it reweights training points to focus on disagreement.

In the Probably Approximately Correct (PAC) framework, AdaBoost guarantees low error given access to a weak learner with fixed advantage over chance.
## Conclusion

You’ve:
- Derived formulas  
- Understood boosting intuition  
- Built AdaBoost from scratch  
- Benchmarked and stress-tested  
- Understood theoretical and empirical behavior

➡️ Ready for LogitBoost, RealBoost, or GBDT.



## 🔍 Why Was AdaBoost Not Enough?

AdaBoost was an elegant idea—treat weak learners as voters, reweight misclassified points, and iterate. But it **hit a wall** once we tried using it on serious, real-world datasets. Here’s why:

---

### **1. Fragility to Noise**

- AdaBoost focuses increasingly on misclassified points.
    
- If those are mislabeled or noisy (which real-world data always is), it **amplifies garbage**.
    
- There's no mechanism to say, _“maybe this data point is too unreliable to trust.”_
    
- It overfits hard, especially on small or noisy datasets.
    

➡️ **No robustness built in**.

---

### **2. Lack of Flexibility in Loss Functions**

- AdaBoost is tied to exponential loss.
    
- This loss function penalizes wrong predictions **too aggressively**, causing sharp gradients.
    
- You can't easily switch to logistic loss, hinge loss, or custom metrics.
    

➡️ **One-size-fits-all objective function** is not ideal when your business or research problem needs control.

---

### **3. No Regularization**

- It doesn’t penalize complex trees.
    
- Every new learner adds more capacity without constraint.
    
- It keeps fitting until it overfits—nothing to stop complexity growth.
    

➡️ There’s **no brake pedal**, just an accelerator.

---

### **4. Weak Base Learners**

- AdaBoost traditionally uses **decision stumps** (1-split trees).
    
- If you increase depth to capture structure, you make it even more prone to overfitting.
    

➡️ It lacks _smart, data-aware trees_ that can grow unevenly and adjust capacity per leaf.

---

### **5. No Efficient Handling of Sparse / Missing Data**

- AdaBoost assumes complete, clean input.
    
- Any missing or sparse representation needs manual cleanup, which breaks pipelines.
    

➡️ Not suited for large-scale, real-world tabular data.

---

### **6. No Use of Second-Order Information**

- AdaBoost optimizes using just first-order error feedback (classification error rate).
    
- It doesn't leverage curvature (Hessian) of the loss landscape, which slows convergence and weakens performance.
    

➡️ **Less informed updates** per iteration.

---

### **7. No Parallelism**

- It's inherently sequential.
    
- You can’t easily speed up learning on modern CPUs or GPUs.
    
- For datasets with millions of rows and columns, it’s a bottleneck.
    

➡️ **Not scalable**.

---

## 🔧 What the Successor Did Better (Strategically)

This newer booster came into play to **address all of the above in one strike**:

|Limitation|What the successor introduced|
|---|---|
|Overfitting|Regularization on both tree size and leaf weights|
|Loss inflexibility|Generalized to arbitrary differentiable losses|
|Noise sensitivity|Doesn’t over-focus on misclassified points—uses gradients, not hard weights|
|Poor scalability|Parallel split finding, histogram binning, and CPU/GPU support|
|Weak base models|Learns deep, asymmetric trees with optimized structure|
|Sparse/missing data|Native handling without preprocessing|
|Naive optimization|Uses second-order derivatives (Hessian) for more accurate updates|


The Sucessor was XGBoost. Here is what XGBoost does:

# XGBoost

