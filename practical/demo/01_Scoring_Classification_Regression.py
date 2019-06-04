#!/usr/bin/env python

# -------------------------------------------------------------------------------------------------------------------
# Scoring binary classifiers using accuracy, precision, and recall
# Ground truth class labels are generated that are either zeros or ones using NumPy's random number generator (RNG)
# -------------------------------------------------------------------------------------------------------------------

import numpy as np

# Some believe the answer to life is within number 42
# Fix the seed of the RNG:
np.random.seed(42)

# Create mock-up dataset: Generate five random labels that are either zeros or ones by picking random integers in the
# range (0, 2).
# These y_true labels are known as the ground truth or true/target value of a data point,that the classifier aims to predict
# Image processing often requires visual inspection, rather than an expert manually labelling ground truth data

y_true = np.random.randint(0, 2, size=5)
# print("Ground Truth Data",y_true)

# Assume a dumb classifier always predicts label 1:

y_predict = np.ones(5, dtype=np.int32)
# print("Predicted Data\t   ",y_predict)

#Correctly predicted only the 2nd data point (true label = 1). All other data points true label = 0. Dumb classifier predicted 1.

# A naive implementation of the accuracy metric using array summation of boolean and dividing the integer result by 
# total number of truth labels:

# print(np.sum(y_true == y_predict) / len(y_true))

# A smarter, and more convenient, implementation is provided by the scikit-learn module "metrics":
from sklearn import metrics

# print(metrics.accuracy_score(y_true, y_predict))

# Define when a ground-truth label and a predicted label is positive (when == 1)

truly_a_positive = (y_true == 1)
# print(truly_a_positive)
predicted_a_positive = (y_predict == 1)
# print(predicted_a_positive)
# Calculate tp, tn, fp, fn by using array summation of booleans

# True positive: Predicted it was a 1, and it (the true label) actually was a 1
true_positive = np.sum(predicted_a_positive * truly_a_positive)
# print("tp =",true_positive)

# Conversely, false positive: Predicted it was a 1, but it was actually a 0
false_positive = np.sum((y_predict == 1) * (y_true == 0))
# print("fp =",false_positive)

# Predicted it was a 0, but it actually was a 1 (this dumb classifier never predicted 0)
false_negative = np.sum((y_predict == 0) * (y_true == 1))
# print("fn =",false_negative)

# Predicted it was a 0, and it actually was a 0
true_negative = np.sum((y_predict == 0) * (y_true == 0))
# print("tn =",true_negative)

# Calculate accuracy again: (everything predicted correctly) divided by (total number of data points):
accuracy = np.sum(true_positive + true_negative) / len(y_true)
# print(accuracy)

# Calculate precision: (true positives) divided by (what the classifier thinks are positives):
precision = np.sum(true_positive) / np.sum(true_positive + false_positive)
# print(precision)
metrics.precision_score(y_true, y_predict) #the easy way

# Calculate recall: (true positives) divided by (what the classifier predicted as positive and did not miss)
# In other words, the fraction of all positives that are correctly classified as positives
# false_negative != 0 means that we missed some positive data for e.g. cats

recall = true_positive / (true_positive + false_negative)
# print(recall)
metrics.recall_score(y_true, y_predict) #the easy way

# Of course recall is 100% since all predictions were positive (but plagued accuracy and precision with  many false predictions)


# -------------------------------------------------------------------------------------------------------------------
# Next, scoring regressors using mean squared error (mse), explained variance, and R squared.
# These three metrics measure how close the predicted labels are to the regression line of the true labels
# -------------------------------------------------------------------------------------------------------------------

# Create a new mock-up dataset: Assume there is data with a "best-fit" that looks like a sine wave.
# Generate 100 equally spaced x values between 0 and 10:

# In[19]:


x = np.linspace(0, 10, 100)


# Real data is often noisy. Create a noisy y_true function by adding random noise to the sin function
# Jitter noise every data point either up or down by a maximum of 0.5:
y_true = np.sin(x) + np.random.rand(x.size) - 0.5

# Matplotlib to visualize data:

import matplotlib.pyplot as plt
from matplotlib import interactive
plt.style.use('ggplot')

plt.figure(figsize=(10, 6))
# plt.plot(x, y_predict, linewidth=4, label='model')
plt.plot(x, y_true, 'o', label='data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower left')
plt.show()
# plt.savefig("sine.png")

# Assume a clever model that figured out that it is a sine wave:
y_predict = np.sin(x)

# MSE is a simply metric to determine how good our model predictions are (by looking at how bad it is). For each data
# point,  the difference between  the predicted and the actual `y` value is calculated and then squared. Lastly,
# calculate the average of the result over all the data points.

# MSE is thus calculated as follows:
# mse = np.mean((y_true - y_predict) ** 2)

# Simply use scikit-learn's built in MSE function instead:

mse = metrics.mean_squared_error(y_true, y_predict)

print("MSE =",mse)

# Explained variance measures the scatter in the data. e.g. no scatter would be if every data
# point was equal to the mean of all the data points allowing perfect predictions of all future data points using a
# single data point the noisy part of y_true can be explained.

# Calculate the variance that exists between the predicted and ground truth to get the "fraction of variance explained":
# ve = 1.0 - (np.var(y_true - y_predict) / np.var(y_true))

# Again scikit-learn's built in function is used instead:
ve = metrics.explained_variance_score(y_true, y_predict)
print("Fraction of Variance Explained",ve)

# The coefficient of determination (R-squared) compares the mean squared error calculated earlier to the actual
# variance in the data:
# r2 = 1.0 - mse / np.var(y_true)

# scikit-learn's built in function:
r2 = metrics.r2_score(y_true, y_predict)
print("R-squared",r2)
