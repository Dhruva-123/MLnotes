
In chapter 3 of Rashka, we will be looking at popular classification algorithms (Logistic regression, support vector machines, decision trees). We will also learn Scikit-Learn in all of its glory.

**Scikit Learn essentials:**

**1.Train Test Splits:**

With the Scikit-Learn library, we can split any given data into train test splits. The code goes as follows:

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X , Y , test_size = 0.3, random_state = 1,  stratify = y)

What we did here is,
1. We took the dataset combo X, Y
2. We are spliting it such that 30 percent is test and 70 percent is train
3. random_state = 1, because, as long as the instance is alive and not closed, the split data is the same. doesn't split differently each time we run it. 
4. And stratify means, if there are 5 different classes in y, we split the data such that each split part gets equal number of classes in each split. ex: say we have a flowers set. There are 5 flowers and 100 samples. we are asking the split to split the data 80 percent train and 20 percent test. then, the training data will split the data as follows: 16 flowers of each kind in train and 4 flowers of each kind to test. stratify with y because we care about y here, (the type of flower).
  
**2.Data Preprocessing:**

As seen in the previous chapters, we sometimes need to standardize the features. That can be done with SciKit-Learn easily. Here we will show the standardization method only but there are a lot more options in feature preprocessing:

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

What we did here goes as follows:
1. imported StandardScalar
2. we made the sc object find the mean and deviation from X_train(always X_train because that is unique to a trained model)
3. we now updated the training features with the known mean and deviation
4. we now updated the test features with the known mean and deviation because, as discussed in the past chapter, the mean and deviation of the train is only used.

3.Models:

Just like our packaged perceptron and adaline and adalineSGD, there are a shit ton of linear classifiers in sklearn library. Just import and use them. The standard perceptron is also available there. Usually, each model is builtin with an accuracy tester called "score" method. use that.

4.We can also use sklearn to plot decision regions. We will get to that soon enough.


**LOGISTIC REGRESSION:**


Now, Here is the idea of what a logistic regression is:
1. Logistic Regression, though named as a regression, is actually a classification algorithm
2. This particular model is based on probability rather than determinate values.
3. This is used for binary classification. Therefore, the outputs are either 1 or 0
4. Here, let p be the probability of X = 1, odds = $p/(1 - p)$ 
5. $logit(X = 1|p) = log(p/(1 - p))$ . Note that this log is the natural log and almost always, we work with natural logs.
6. Now, what this $logit(X = 1|p)$ does is, it takes a random probability from [0,1] and converts it into a real number on the number line.
7. The whole idea of logistic regression is to take a real number, net_input, and convert it into a probability in the range of [0 , 1]. That being the case, we can clearly see that the inverse of the logit function gives us what we need.
8. That inverse is called "Logistic Sigmoid Function". and it is represented by $\phi(z)$ where $z$ is the net_input(dot product of weights and features). The function goes as follows:
9. $\phi(z) = \frac{1}{1 + e^{-z}}$ where $z$ is the net input. Now, when we plot this function, the graph would usually be in the shape of an S. An image will be attached to this text to represent the graph and its intercepts.
   
   ![[Pasted image 20250620161322.png]]
10. In the case of adaline, we used the activation function $\Phi(z)$ to be the indentity function. Here, we will be using the above mentioned sigmoid function as the activation function instead.
11. Once the net_input passes through the sigmoid function, we will be getting a value between 0 and 1. we will then classify the give data based on the output, wether the resulted output from the sigmoid function is greater than 0.5 or not would be the metric. If the output is greater than 0.5, then we declare the example to be of the class X = 1, otherwise, X = 0 is the class that example goes under.

The same thing can be done in sklearn from the following code example:
This code is for multiple classification:

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 100.0 , random_state = 1, solver = "lbfgs" , multiclass = "ovr") ### Multiclass = ovr because oVr is a technique for multiple classifications using the same model. Now, There are other methods like multinomial.###
lr.fit(X_train_std, Y_train)


That's it. Limited Memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) is a memory efficient method for the multinomial case, that's why we are using it here.

From our own implementation of multinomial logisitic regression, we have found several things:

Multinomial logisitic regression depends on several components. we have softmax function, one_hot function. What we are essentially doing there is this:
1. Take the input data and train on it. Once trained, we then try to predict the output via heavy probability. This is where to softmax function comes in. Softmax function gives a probability distribution for a given net_input z. That probability distribution tells us the chances of a particular example being one of n classes. we then predict the class that has the highest probability. This probability formation is done fairly simply with softmax function. 
2. For error calculations and weight updates, we use the gradient of the loss function of the multinomial case. the loss function of the multinomial case looks something like this:
3. $\mathcal{L} = - \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} \cdot \log(\hat{y}_{i,k})$
   Where i stands for the $i^{th}$ example and k stands for the $k^{th}$ feature. 
4. we find the gradient to this loss function wrt weights and we get the error fixes or $\Delta W$ for each iteration.

Overfitting is a term used to describe a model that performs well on training data but fails to work well for test data. This is due to too many parameters, too many epochs, etc

Similarly, underfitting is too little epochs or parameters for the model to truly grasp the different nuances in the data.

NOTE:
"High variance" is proportional to over fitting and "High bias" is proportional to underfitting.
**Bias** is the error introduced by approximating a real-world problem, which may be complex, by a **simplified model**
**Variance** is the error introduced by the model’s **sensitivity to small fluctuations** in the training data.


**REGULARIZATION:**

This is a little tweak that we do to the loss function that makes sure that we don't overfit easily. Regularization is the concept of adding a term that pulls the loss or the gradient based change in weights away from the actual change. It is like friction, when the force (error) is strong enough to bypass friction (regularization term), then, the object moves (The update gets made). There are a lot of different regularization terms in the market but for now, we will be talking about L1 and L2 regularization terms.

**L2 Regularization term:**


$\frac{\lambda}{2} \|\mathbf{w}\|^2 = \frac{\lambda}{2} \sum w_i^2$

That is the formula for L2 regularization. 
$\frac{\lambda}{2} \sum w_i^2$ is the term that we are adding to the loss function in each of the models. However, the loss function is derived wrt W to be added to W in each iteration. so, The derived term would simply be $\lambda W$ . We usually have lambda to be a very small value, just like eta, cuz think, it makes sense.

**L1 Regularization term:**

$\lambda\sum W$ is the regularization term added to the loss function. This is a bit more mathematically nuanced, so, L2 term is usually used. That's why we are only going to cover L2 term for now.


**SUPPORT VECTOR MACHINES (SVM):**

Support vector machines is a method to optimize the creation and adjustment of a hyperplane that seperates two different classes of objects. Think of this as an upgrade to perceptron.

say we have two different classes and they are labeled -1 and 1 respectively. How whole idea with the perceptron is $W^TX + B$ > 0 , then class 1, or < 0 , class -1. But here, we will create two margins, $W^TX + B = 1$ , $W^TX + B = -1$ . These two margins show exactly where the class boundary of class 1 and class -1 are. If the object is between these margins, we consider the object as unknown and adjust the hyperplane such that the object is correctly classified and not inside the margin. ofc,  we can't do this for every class, therefore, we only do it to linearly seperable classes.

let $\|\mathbf{w}\|$  be the sum of squares of every $w_i$ inside w_. we find that the margin is given by;

margin = 2/$\|\mathbf{w}\|$ . Our whole intention of this support vector machine math is to maximize the margin. Because geometrically, the greater the margin, the better the classification. Therefore, we try to minimize the $\|\mathbf{w}\|$/2 with gradient descent. Let me show you how this goes;

say for example 1;
$W_0$ + $W^TX$ > 1, This means that the object that we are checking rn is well above the margin and correctly classified. That means, we dont need to change the hyperplane at all.

say for example 2;
0 < $W_0$ + $W^TX$ < 1, This means that, even thought the object is correctly classified (above zero), the object is inside the margin. Therefore, we have error and we will change the hyperplane.

say for example 3:

$W_0$ + $W^TX$  < 0, Here, the object is clearly missclassified. That means, we have to adjust the hyperplane in both directional and orientation ways.

This is good in theory, however, when we are actually practicing real world ML, These strict margins lead to a lot of missclassifications and margin errors. Just like with regularization, we will try to introduce a new variable that tells us how wrong we are, not just if we are right or wrong. This allows us to make changes that are relative to the degree of mistake that we are making. These variables are called slack variables.

Slack Variables:
$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i
$$
The above equation is that we are trying to introduce into the weights are error. The new C and and $\xi$ are the slack variables here. Note that all $\xi_i$ are greater than 0. C is called the panelty parameter.










