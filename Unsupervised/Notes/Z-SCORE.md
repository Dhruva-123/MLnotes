
## 1. Overview

**Z-score** is a **standardization technique** in statistics used to describe the relative position of a data point within a dataset. It tells you **how many standard deviations** a value is from the mean.

- It is commonly used in **outlier detection**, **normalization**, and **statistical comparison**.
    
- Works best when the data is approximately **normally distributed**, but can still be used as a relative measure.
    

---

## 2. Formula

The **z-score** of a value $xix_ixi$​ in a dataset is defined as:

$zi=xi−μσz_i = \frac{x_i - \mu}{\sigma}zi​=σxi​−μ$​

Where:

| Symbol     | Meaning                                           |
| ---------- | ------------------------------------------------- |
| $xix_ixi​$ | Value of the data point                           |
| $μ\muμ$    | Mean of the dataset                               |
| $σ\sigmaσ$ | Standard deviation of the dataset                 |
| $ziz_izi​$ | Z-score (number of standard deviations from mean) |

---

## 3. Intuition

- zi=0z_i = 0zi​=0 → the value is **exactly at the mean**.
    
- zi>0z_i > 0zi​>0 → the value is **above the mean**.
    
- zi<0z_i < 0zi​<0 → the value is **below the mean**.
    
- ∣zi∣>3|z_i| > 3∣zi​∣>3 → typically considered an **outlier** (for normal distributions).
    

---

## 4. Step-by-Step Calculation

Suppose you have the dataset:

X=[10,12,15,18,20]X = [10, 12, 15, 18, 20]X=[10,12,15,18,20]

**Step 1: Compute the mean**

$μ=10+12+15+18+205=15\mu = \frac{10 + 12 + 15 + 18 + 20}{5} = 15μ=510+12+15+18+20​=15$

**Step 2: Compute standard deviation**

$σ=(10−15)2+(12−15)2+(15−15)2+(18−15)2+(20−15)25\sigma = \sqrt{\frac{(10-15)^2 + (12-15)^2 + (15-15)^2 + (18-15)^2 + (20-15)^2}{5}}σ=5(10−15)2+(12−15)2+(15−15)2+(18−15)2+(20−15)2​​$
$σ=25+9+0+9+255=13.6≈3.686\sigma = \sqrt{\frac{25 + 9 + 0 + 9 + 25}{5}} = \sqrt{13.6} \approx 3.686σ=525+9+0+9+25​​=13.6​≈3.686$

**Step 3: Compute z-scores**

$z1=10−153.686≈−1.36z_1 = \frac{10 - 15}{3.686} \approx -1.36z1​=3.68610−15​≈−1.36$

$z2=12−153.686≈−0.81z_2 = \frac{12 - 15}{3.686} \approx -0.81z2​=3.68612−15​≈−0.81$

…and so on for all points.

## 6. Applications of Z-Score

|Application|Description|
|---|---|
|Outlier Detection|Points with|
|Standardization / Normalization|Transform features to have mean 0 and std 1 for ML models.|
|Comparison Across Datasets|Allows comparison of scores from different distributions.|
|Anomaly Detection|Detect unusual patterns in data streams or logs.|

---

## 7. Related Simple Statistical Methods

| Method                        | Formula / Definition                                                                         | Use Case                                 |
| ----------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Mean (μ\muμ)                  | $∑xin\frac{\sum x_i}{n}n∑xi​​$                                                               | Central tendency                         |
| Median                        | Middle value of sorted data                                                                  | Central tendency, robust to outliers     |
| Standard Deviation (σ\sigmaσ) | $∑(xi−μ)2n\sqrt{\frac{\sum (x_i - \mu)^2}{n}}n∑(xi​−μ)2​​$                                   | Spread / dispersion                      |
| Variance (σ2\sigma^2σ2)       | $∑(xi−μ)2n\frac{\sum (x_i - \mu)^2}{n}n∑(xi​−μ)2​$                                           | Square of std, measure of spread         |
| Interquartile Range (IQR)     | Q3−Q1Q3 - Q1Q3−Q1                                                                            | Spread of middle 50%, robust to outliers |
| Min-Max Normalization         | $x′=x−xmin⁡xmax⁡−xmin⁡x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$x′=xmax​−xmin​x−xmin​​$ | Scale between 0–1                        |
|                               |                                                                                              |                                          |
