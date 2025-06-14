**Supervised Learning:**
In supervised learning, our main goal is to teach a model with labeled data that allows us to make predictions about unseen or future data. 

there are two main subcategories of supervised learning:
1. Classification (We will segregate a given data into different classes (say cat, dog, pig etc when given a set of animals))
2. Regression (Here, the outcome is a continuous value)

    Classification:
     In classification, we can have any number of classes that we are segregating the given data into, and the model must segregate them with a certain accuracy. Let us consider a case like this:
         We have a dataset consisting of two types of data, say "-ve data" and "+ve data". There are two variables or parameters that govern if a given data point is +ve or -ve.
         Say these parameters are x1, x2. In supervised learning, The model must find that dotted line and segregate the two classes based on the position of the point wrt the line. We train the model to find that line and once it does, we will have our answer with varying accuracy ofc.
         ![[Pasted image 20250614114259.png]] 
         This is the graph which helps us visualize the given question but in real world, the parameters will be in the hundreds and they maybe independent on one another or not. The cases will be more complex, ofc, but the general idea is being conveyed.
     In the general case, however, we may have multiclass classification. The most common example is predicting the given alphabet in hand written notes. This considers every single alphabet and makes a prediction later. This is not a binary classification like the above given example.

	Regression:
	    In regression, there are a number of predictor variables (Those variables that effect the outcome) and a response variable (The actual outcome) . Our job in regression is to find the relationship between the given predictor variables and the response variables. Usually, the predictor variables are called "Features" while the outcome is called "Target Variable".
	        Example:
	         Here, say we have a feature 'Y' that is related to the target variable 'X'. Here is a simple demonstration of the regression process.
	         ![[Pasted image 20250614115428.png]]
	         Our intention here is to draw a line such that the squared distances from the points to the line is is minimal. Then, we will base the relationship of the variables Y and X on this line and make predictions based on that line that we created based on the training data.

    Reinforcement Learning:
        Reinforcement learning, although not a branch of supervised learning, follows the general path of the supervised learning map. Here, an agent is deployed into the environment, whose actions trigger a reward or punishment function that scores the agent based on it's performance and based on the results obtained. 


Unsupervised Learning:
    Here, the general idea is to explore a given unknown or undefined data and  bring out insights from the data given without a reward system or prior labeling.
     There are two fundamental use cases for unsupervised learning in the ML pipeline:
        1. Finding subgroups with clustering:
            Say we give a pile of data. This unsupervised learning technique groups these data points and classifies them based on their relationship with each other based on the given metrics we measure against. This is called unsupervised classification. In the following diagram, we will see different groups forming clusters when we consider two parameters x1 and x2.
             ![[Pasted image 20250614124803.png]]
         2. Dimensionality Reduction For Data Compression:
            This method is a commonly used method in feature preprocessing to remove noise from the data, and compress the data to a lower dimensional subspace. This makes ML computationally feasable. This is a general technique that we use preprocessing of data before feeding it to any ML algorithm.



Some terms to bare in mind:
    Signal means usefulness. There is a signal to noise ratio, which means useful to useless ratio