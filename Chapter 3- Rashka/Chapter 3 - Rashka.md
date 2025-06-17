
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