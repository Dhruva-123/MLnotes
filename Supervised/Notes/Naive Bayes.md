# 🧠 Naive Bayes — Deep-Dive Notes

---

## I. 🧭 Conceptual Foundations

### 🔸 What is Naive Bayes?

- A **generative classification algorithm**.
    
- Uses **Bayes’ theorem** + **strong (naive) feature independence** assumptions.
    
- Efficient, scalable, and surprisingly effective on many high-dimensional tasks (like text classification).
    

---

### 🔸 Bayes' Theorem

$$P(y∣x)=P(y)P(x∣y)P(x)P(y \mid \mathbf{x}) = \frac{P(y) P(\mathbf{x} \mid y)}{P(\mathbf{x})}P(y∣x)=P(x)P(y)P(x∣y)​$$

For classification:

- We ignore the denominator (same for all classes)
    

$P(y∣x)∝P(y)P(x∣y)P(y \mid \mathbf{x}) \propto P(y) P(\mathbf{x} \mid y)P(y∣x)∝P(y)P(x∣y)$

We take **log** to avoid numerical underflow:

$$log⁡P(y∣x)∝log⁡P(y)+log⁡P(x∣y)\log P(y \mid \mathbf{x}) \propto \log P(y) + \log P(\mathbf{x} \mid y)logP(y∣x)∝logP(y)+logP(x∣y)$$

---

### 🔸 Naive Assumption

Assume features are conditionally independent given the class:

$$P(x∣y)=∏j=1nP(xj∣y)P(\mathbf{x} \mid y) = \prod_{j=1}^{n} P(x_j \mid y)P(x∣y)=j=1∏n​P(xj​∣y) log⁡P(x∣y)=∑j=1nlog⁡P(xj∣y)\log P(\mathbf{x} \mid y) = \sum_{j=1}^{n} \log P(x_j \mid y)logP(x∣y)=j=1∑n​logP(xj​∣y)$$

This drastically simplifies likelihood estimation.

---

### 🔸 Probability vs Likelihood

| Term                                         | Meaning | In Naive Bayes                              |
| -------------------------------------------- | ------- | ------------------------------------------- |
| **Probability**                              | P(y     | x) — Posterior                              |
| **Likelihood**                               | P(x     | y) — Treat data as fixed, class as variable |
| Bayes Rule flips likelihood into probability |         |                                             |

---

## II. 🧩 Model Structure (Common to All NB Variants)

python

CopyEdit

`class NaiveBayes:     def fit(self, X, y): ...     def joint_log_likelihood(self, X): ...     def predict(self, X): ...`

- `fit()` estimates prior + likelihood terms.
    
- `joint_log_likelihood(X)` returns $$log⁡P(y)+log⁡P(x∣y)\log P(y) + \log P(\mathbf{x} \mid y)logP(y)+logP(x∣y)$$ for each class.
    
- `predict()` selects class with highest log-probability.
    

---

## III. 📘 Multinomial Naive Bayes

---

### 📌 Use Case

- **Text classification**, word-count features (bag-of-words, n-grams).
    
- Input: non-negative integers (word counts, term frequencies).
    

---

### 🧮 Likelihood Estimation

For class y=cy = cy=c, the likelihood of feature xjx_jxj​ is:

$$θcj=P(xj∣y=c)=Ncj+α∑j(Ncj+α)\theta_{cj} = P(x_j \mid y = c) = \frac{N_{cj} + \alpha}{\sum_j (N_{cj} + \alpha)}θcj​=P(xj​∣y=c)=∑j​(Ncj​+α)Ncj​+α​$$

Where:

- $NcjN_{cj}Ncj$​: Total count of feature jjj in all samples of class ccc
    
- α\alphaα: Smoothing constant (Laplace: α = 1)
    

---

### 💡 Log-Joint Likelihood

$$log⁡P(y=c∣x)∝log⁡P(y=c)+∑j=1nxjlog⁡θcj\log P(y = c \mid \mathbf{x}) \propto \log P(y = c) + \sum_{j=1}^n x_j \log \theta_{cj}logP(y=c∣x)∝logP(y=c)+j=1∑n​xj​logθcj​$$

This becomes:

python

CopyEdit

`X @ self.feature_log_prob_ + self.class_log_prior_`

Shape: `(n_samples, n_classes)`

---

### 🧠 Intuition

- The log-likelihood is **dot product** of word-counts and log probabilities of words given class.
    
- Multinomial distribution means: count how often each word appears, and treat each occurrence as an independent draw from a vocabulary-specific distribution.
    

---

## IV. 📗 Gaussian Naive Bayes

---

### 📌 Use Case

- Continuous numerical features (e.g., sensor data, pixel intensities).
    
- Assumes **features are normally distributed** per class.
    

---

### 📐 Likelihood Model

Each feature xjx_jxj​ is modeled with a class-specific Gaussian:

$$P(xj∣y=c)=12πσcj2exp⁡(−(xj−μcj)22σcj2)P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} \right)P(xj​∣y=c)=2πσcj2​​1​exp(−2σcj2​(xj​−μcj​)2​)$$

---

### 💡 Log-Joint Likelihood

After summing over features:

$$log⁡P(y=c∣x)∝log⁡P(y=c)−12∑j[log⁡(2πσcj2)+(xj−μcj)2σcj2]\log P(y = c \mid \mathbf{x}) \propto \log P(y = c) - \frac{1}{2} \sum_j \left[ \log(2\pi \sigma_{cj}^2) + \frac{(x_j - \mu_{cj})^2}{\sigma_{cj}^2} \right]logP(y=c∣x)∝logP(y=c)−21​j∑​[log(2πσcj2​)+σcj2​(xj​−μcj​)2​]
$$
---

### ✳️ Log-Likelihood Vectorized

python

CopyEdit

`log_likelihood = -0.5 * np.sum(     np.log(2. * np.pi * self.var_) + ((X[:, np.newaxis, :] - self.mean_) ** 2) / self.var_,     axis=2 )`

**Shape flow:**

- `X[:, np.newaxis, :]` → shape: (n_samples, 1, n_features)
    
- `mean_` → (n_classes, n_features)
    
- Result → broadcasted to (n_samples, n_classes, n_features)
    
- `sum(axis=2)` → final log-likelihood shape: (n_samples, n_classes)
    

Add priors:

python

CopyEdit

`return log_likelihood + self.class_log_prior_`

---

## V. 🧠 General Design Pattern

- All Naive Bayes variants inherit from a `NaiveBayes` base class
    
- Each variant implements `.fit()` and `.joint_log_likelihood(X)`
    
- Base `.predict()` uses:
    
    python
    
    CopyEdit
    
    `table = self.joint_log_likelihood(X) return self.classes_[np.argmax(table, axis=1)]`
    

Optional:

python

CopyEdit

`def predict_proba(self, X):     log_probs = self.joint_log_likelihood(X)     log_probs -= log_probs.max(axis=1, keepdims=True)     probs = np.exp(log_probs)     return probs / probs.sum(axis=1, keepdims=True)`

---

## VI. 🧪 Comparison Summary

|Variant|Feature Type|Model|Use Case|
|---|---|---|---|
|**Multinomial**|Discrete counts|Multinomial|Text classification|
|**Gaussian**|Continuous|Normal distribution|Tabular/numerical sensor data|
|**Bernoulli**|Binary|Bernoulli distribution|Binary word presence (next)|

---

## VII. 🔥 You’ve Implemented

- Base class: prediction logic
    
- Full Multinomial Naive Bayes:
    
    - With Laplace smoothing
        
    - Joint log-likelihood vectorized
        
- Full Gaussian Naive Bayes:
    
    - With numerical stability (`+1e-9`)
        
    - Matrix-broadcasted log-likelihood with `axis=2`
        
- Understood dimensions, formulas, log-transform benefits, and independence assumptions