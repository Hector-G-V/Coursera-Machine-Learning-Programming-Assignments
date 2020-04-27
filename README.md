# Coursera-Machine-Learning-Programming-Assignments
Training machine learning models using the data in the Andrew Ng, Stanford University Coursera course.

## Example 1
<p>
<i>Example 1.py</i> uses the <i>LinearRegression</i> function in the <i>sklearn.linear_model</i> module to fit input data and print the model weight(s) and bias term.
</p>

### Input
The file name may be input from the command line. 
```
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file', default='ex1data1.txt', type=str,  help='file name')
```
<p>
The default <i>ex1data1.txt</i> is one of two data sets provided with this assignment. It is found in the Example 1 <i>Data</i> folder, along with the other assignment-provided set, <i>ex1data2.txt</i>
</p>

### '__main__' Output
The module prints the weight(s) and bias term that fit the specified data, along with the Mean Square Error for comparison to other algorithms.


### Input Formatting
<p>
To use data sets that are not provided with the assignment, the data must be arranged in the <i>.txt</i> file in a specific way: (i) data for target and features must be in columns. (ii) the last column must be the target data.
</p>

## Example 2
<p>
<i>Example 2.py</i> uses the <i>LogisticRegression</i> function in the <i>sklearn.linear_model</i> module to train a model on a 2-feature input data set and plot the classified points with the decision boundary.
</p>

### Input
<p>
The file name may be input from the command line using the flag <i>-file</i>. The data is split into a testing and training set.

The default <i>ex2data1.txt</i> is one of two data sets provided with this assignment. It is found in the Example 2 <i>Data</i> folder, along with the other assignment-provided set, <i>ex2data2.txt</i>
</p>

### '__main__' Output
Prints the prediction accuracy for both the training and testing data sets.

Plots all the data points and the decision boundary.


### Input Formatting
<p>
To use data sets that are not provided with the assignment, the data must be arranged in the <i>.txt</i> file in a specific way: (i) data for target and features must be in columns. (ii) the last column must be the target data.
</p>

Two parameters may be used to improve the prediction accuracy. These parameters may the changed fromt he command line. The flags are: inverse of regularization strength: <i>-C</i>; mapped features polynomial degree: <i>-d</i>.

Note: appropriate <i>-step</i> values are <i>0.5</i> for <i>'ex2data1.txt'</i>, and <i>0.01</i> for <i>'ex2data2.txt'</i>.

## Example 3
<p>
<i>Example 3.py</i> uses the <i>LogisticRegression</i> function in the <i>sklearn.linear_model</i> module to train a model on a set of handwritten digit images. The algorithm is one-vs-rest. 
  
The code is nearly identical to the module <i>Example 2.py</i>. The only difference is specification of the parameter <i>multi_class='ovr'</i>, for one-vs-rest.
</p>

### Input
<p>
The file name may be input from the command line using the flag <i>-file</i>. The data is split into a testing and training set.

The default <i>ex3data1.mat</i> is one of two data sets provided with this assignment. It is found in the Example 3 <i>Data</i> folder.
</p>

### '__main__' Output
Prints the prediction accuracy for both the training and testing data sets.

### Input Formatting
<p>
To use data sets that are not provided with the assignment, the data must be arranged in rows, where one row contains all the pixel values for one image.
</p>

Inverse of regularization strength <i>C</i> may be changed from the command line to improve the prediction accuracy. The flag is <i>-C</i>, and the default value is <i>C=1</i>.

## Example 4

Feed-Forward Neural Network with one hidden layer is trained with TensorFlow tools.

Note: using rectified linear activation for the hidden layer. The assignment uses sigmoid activation.

### Input
<p>
The file name may be input from the command line using the flag <i>-file</i>. The default file <i>ex4data1.mat</i> is provided with this assignment. It is found in the Example 4 <i>Data</i> folder.
</p>

### '__main__' Output
Prints the prediction accuracy for the training data sets.

### Input Formatting
<p>
To use data sets that are not provided with the assignment, the data must be arranged in rows, where one row contains all the pixel values for one image.
</p>

Two parameters may be used to improve the prediction accuracy. These parameters may the changed fromt he command line. The flags are: number of nodes in the hidden layer: <i>-n</i>; number of training epochs: <i>-epochs</i>.

## Example 5
<i>Example 5.py</i> uses an sklearn linear regression function to fit test data with polynomial-mapped features, and plots the error as a function of the specified parameter. The parameter options are: test size, polynomial degree, and regularization parameter.

### Input

The file name may be input from the command line using the flag <i>-file</i>. The data contains testing, training and cross-validation sets.

The default <i>ex5data1.mat</i> is found in the Example 5 <i>Data</i> folder.

### '__main__' Output
Three plots: error as a function of test set size; error as a function of polynomial degree, and error as a function of the regularization parameter.

### Input Formatting

The regularization parameter <i>alpha</i> may be changed from the command line to improve the prediction accuracy. The flag is <i>-alpha</i>, and the default value is <i>alpha=568</i>.

The polynomial degree <i>degree</i> may be changed from the command line to improve the prediction accuracy. The flag is <i>-degree</i>, and the default value is <i>degree=3</i>.

## Example 6

<i>Example 6.py</i> uses the <i>SVM</i> function in the <i>sklearn</i> module to train a model to classify data and to classify spam.

### Input

The file name may be input from the command line using the flag <i>-file</i>. The data is split into a testing and training set.

This assignment did not provide classified, raw email data for pre-processing AND testing. Therefore, the file <i>spam.csv</i> is included. The file is a dataset of classified, raw email data that may be used to train a model with this module.

For sake of generality to most data, input data is automatically splits to test and train. Therefore, this module is unable to train with <i>spamTrain</i> and test with <i>spamTest</i> without changing the <i>main()</i> function.


### '__main__' Output
Prints the prediction accuracy for both the training and testing data sets.

Plots if the features data is 2D.

### Input Formatting

The keys for almost all datasets are: <i>X_key: 'X', y_key: 'y'</i>. The exceptions are <i>spamTest.mat</i> (<i>X_key: 'Xtest', y_key: 'ytest'</i>), and <i>spam.csv</i> (<i>X_key: 'EmailText', y_key: 'Label'</i>).

## Example 7

<i>Example 6.py</i> provides the option to run K-Means or PCA on the assignment data. The module is designed to process data with dimensions that are consistent with the assignment.

### Input

The file name may be input from the command line using the flag <i>-file</i>.

The model choice - either PCA or K-Means - is indicated with flag <i>-model</i>.

The number of clusters(K-means) or components (PCA) is indicated with the flag <i>-n</i>.


### '__main__' Output
Plot of the data or image.

If the data is an image: plots the original image and the processed image. If PCA, the image number to plot is indicated with the flag <i>-pca_img</i>.

## Example 8

<i>Example 8.1.py</i> is the anomaly detection module. It fits training data, then prints a range of epsilons and F1 scores for a validation set. 

<i>Example 8.2.py</i> is the recommender system module. It takes data Y, R and finds a features vector X, and ratings vector (weights) Theta. 

### Input

The file name may be input from the command line using the flag <i>-file</i>.

<i>Example 8.1.py</i>: Training and validation data; min and max range for optimal epsilon search.

<i>Example 8.2.py</i>: Training and debugging data; length of features and weights vectors; number of gradient descent iterations.

### '__main__' Output

<i>Example 8.1.py</i>: F1 scores for epsilon values in the indicated range.

<i>Example 8.2.py</i>: Trained X, Theta, and a plot of the cost function over the iterations.
