# Supervised Learning Algorithms:

## There are 2 major types of Supervised learning algorithms that classify given data:

### 1. Perceptron Method:

It is based on human neuron model called MCP neuron. Here is its mathematical implementation.
Say we are given and input vector **x** containing x1, x2, x3... and we create a weight vector **w** containing w1, w2, w3... we then create a output function that is the matrix multiplication of the vectors **x** and **w** and name it **z** and $z = x1w1 + x2w2 + x3w3...$

If the value of z > $\theta$ where $\theta$ is a defined threshold , we classify the given vector into +ve (+1) group, else,  we classify the given vector into -ve (-1) group. $\Phi(z)$ is the function that checks if the output $z$ is greater than $\theta$ or less than it. The output of $\Phi(z)$ gives us the class of the input vector.