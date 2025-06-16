# Supervised Learning Algorithms:

## There are 2 major types of Supervised learning algorithms that classify given data:

### 1. Perceptron Method:

It is based on human neuron model called MCP neuron. Here is its mathematical implementation.
Say we are given and input vector **x** containing x1, x2, x3... and we create a weight vector **w** containing w1, w2, w3... we then create a output function that is the matrix multiplication of the vectors **x** and **w** and name it **z** and $z = x1w1 + x2w2 + x3w3...$

If the value of z > $\theta$ where $\theta$ is a defined threshold , we classify the given vector into +ve (+1) group, else,  we classify the given vector into -ve (-1) group. $\Phi(z)$ is the function that checks if the output $z$ is greater than $\theta$ or less than it. The output of $\Phi(z)$ gives us the class of the input vector.
since $z = x1w1 + x2w2 + x3w3...$ and z > $\theta$ we change the equation like so, by adding another term, $w0x0$ with $x0 = 1$ and $w0$ = -$\theta$ . Now, the new equation is  $z = x0w0 + x1w1 + x2w2 + x3w3...$ 
and we can just check if $z$ > 0. simple.

That checking is done my the function $\Phi(z)$ like so:
𝜙(𝑧) = { 1 if 𝑧 ≥ 𝜃 else -1 } 
Here, $w0$ = -$\theta$ is called the bias unit.

The algorithm for this goes as follows:
1. Initialize the weights to 0 or small random variables.
2. for each training example, $x^i$ :
      a. compute the output value, $\hat{y}$ 
      b. update the weights
updating the weights is done like so;
$w_j := w_j + \Delta w_j$
here $\Delta w_j$ is the change in $w_j$ that we found during the training which is calculated like so;
$\Delta w_j = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_j^{(i)}$
here , $\eta$ is called the learning rate (usually between 0.0 and 1.0) $y^{(i)}$ is the true class or the true answer and $\hat y^{(i)}$ is called the answer that we obtained through the model or predicted class. Here, i stands for the $i^{th}$ training example. that's it. 
Therefore, we calculate the change in w for every w like so;
$\Delta w_0 = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_0^{(i)}$
$\Delta w_1 = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_1^{(i)}$
$\Delta w_2 = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_2^{(i)}$
$\Delta w_3 = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_3^{(i)}$
and so on...
Here, $\eta$ is what we call a hyperparameter. We set it during the perparation of this entire model. For different hyperparameters, we get different outputs. If $\eta$ is too small, the model's speed sucks. If it is too high, the model suffers overshooting and might never converge ever. So the selection must be clever.

This model is only usable if the two classes that we are trying to seperate are linearly seperable and doesn't work if the given data is not seperable.

![[Pasted image 20250614151732.png]]

The general concept of the model is given in the following diagram:

![[Pasted image 20250614151908.png]]

Adaline Algorithm:




