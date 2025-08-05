# Supervised Learning Algorithms:

## There are 2 major types of Supervised learning algorithms that classify given data:

### 1. Perceptron Method:

It is based on human neuron model called MCP neuron. Here is its mathematical implementation.
Say we are given and input vector **x** containing x1, x2, x3... and we create a weight vector **w** containing w1, w2, w3... we then create a output function that is the matrix multiplication of the vectors **x** and **w** and name it **z** and $z = x1w1 + x2w2 + x3w3...$

If the value of z > $\theta$ where $\theta$ is a defined threshold , we classify the given vector into +ve (+1) group, else,  we classify the given vector into -ve (-1) group. $\Phi(z)$ is the function that checks if the output $z$ is greater than $\theta$ or less than it. The output of $\Phi(z)$ gives us the class of the input vector.
since $z = x1w1 + x2w2 + x3w3...$ and z > $\theta$ we change the equation like so, by adding another term, $w0x0$ with $x0 = 1$ and $w0$ = -$\theta$ . Now, the new equation is  $z = x0w0 + x1w1 + x2w2 + x3w3...$ 
and we can just check if $z$ > 0. simple.

That checking is done by the function $\Phi(z)$ like so:
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

Adaline Algorithm: (Short for adaptive linear neuron)

Adaline is very similar to perceptron. But, Adaline has a fundamental property that perceptron doesn't have. In perceptron, we calculate the change with the help of error, and that error is calculated by the prediction and real answer difference. This usually means only 4 cases, 

-1 - (- 1)   -> the answer is -1 and the prediction is -1
 1 - (1)   -> the answer is 1 and the prediction is 1
-1 - (1)   -> the answer is -1 and the prediction is 1
 1 - (- 1)   -> the answer is 1 and the prediction is -1

Although this will mean convergence over time, this can be a bit lacking because, the error doesn't express just HOW wrong the prediction is. 

Adaline fixes that issue, by replacing the error calculation mechanism and also the weight balance mechanisim. First, the error is calculated by the actual output and the answer differences. This means, we will know how wrong the model is for now.
theoritical showcase:
Perceptron  -> $y - \Phi(z)$ = error
Adaline       -> $y - z$ = error 

secondly, the weight balance mechanism goes via gradient descent. to understand gradient descent, lets see what a cost function is:

Cost Function = $J(\mathbf{w}) = \frac{1}{2} \sum_{i} \left( y^{(i)} - \phi(z^{(i)}) \right)^2$ 

Note that here $\phi(z)$ means the activation function. Not to confuse with the prediction value $\Phi(z)$
. Two totally different things. For adaline however, the activation function $\phi(x) = x$
. It's an identity function, but we can change that however we want depending upon the model we are running.

when we say gradient descent, we mean, we are going to calculate the gradient of the cost function wrt the weights, then, we will find the particular weights vector for which the cost function is the least over many iterations. each iteration takes us one step closer to the perfect vector point, but, at the risk of over fitting (means getting too used to the training dataset and being worthless to the real world test datasets). So, keep the iterations in check. When mathematically calculating the gradient of the cost function,  we find that;

    $\nabla J(\mathbf{w}) = - X^\top (\mathbf{y} - \mathbf{z})$

We find that the gradient is $X^{T}$  * (error) and the error here is the adaline error and not the perceptron error. Adaline is coded up and attached to this folder.


**Improving Gradient Descent using feature scaling:**

when we are giving the model real input vectors X, the whole calculation takes time and also causes slow convergence. Therefore, for different models, we follow different principles to make the input vectors X more optimised to converge faster. One of the most standard ways of doing that is by doing "Standardization" of features for gradient descent specific models.


Standardization:

Standardization of a feature in X, say $x_i$ goes as follows:

$x^{'}_i$  = ($x_i - \mu_i$)/$\sigma_i$ 

$\mu_i$ = mean of the features
$\sigma_i$ = standard deviation of the features

Our intention is to make the mean of the new feature to be zero and the standard deviation of the new feature to be 1.

This makes convergence and mathematical calculations easier. Note that when we are using a standardized model, the input features for testing must also be standardized when we want to test them.

Now, we will learn something called Stochastic Gradient Descent:

It's the exact same as adaline, but, in adaline, we send all of the input vectors, X, in at once. This could cause inefficient fitting. That's why, we use SGD. Here, we send each input one by one and that helps the model converge faster. its called adalineSGD


