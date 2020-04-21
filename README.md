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
<i>Example 2.py</i> uses the <i>LogisticRegression</i> function in the <i>sklearn.linear_model</i> module to train a 2-feature input data set and plot the classified points with the decision boundary.
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

Two parameters may be used to improve the prediction accuracy. These parameters may the changed fromt he command line. The flags are:
<pre>
<i>-C</i>: Inverse of regularization strength.</br>
<i>-d</i>: Mapped features polynomial degree.
</pre>

