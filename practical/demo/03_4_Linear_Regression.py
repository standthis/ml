# -------------------------------------------------------------------------------------------------------------------
# Regression is about predicting continuous outcomes rather than discrete class labels.
# Use linear regression to predict Boston housing prices using information such as crime rate,
# property tax rate, distance to employment centers, and highway accessibility.
# -------------------------------------------------------------------------------------------------------------------

# Scikit-learn includes some sample datasets. The Boston dataset is imported:

import numpy as np
import cv2

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})

boston = datasets.load_boston()


# The structure of the boston object:
# - `DESCR`: Get a description of the data
# - `data`: The actual data, <`num_samples` x `num_features`>
# - `feature_names`: The names of the features
# - `target`: The class labels, <`num_samples` x 1>
# - `target_names`: The names of the class labels

# print(dir(boston))

# There are 506 data points; each have 13 feature values:
# print(boston.data.shape)

# Target inspection shows only a single target value
# i.e. label corresponding to the target value of housing price:
# print(boston.target.shape)

# 13 features:
# print(boston.feature_names)

# Since earlier inspection of target values revealed that data is not categorized as classes, it is a regression problem:
# print(np.unique(boston.target))

# Tackle it using Linear Regression
# -------------------------------------------------------------------------------------------------------------------
# Training the model
linreg = linear_model.LinearRegression()
# linreg = linear_model.Ridge()
# linreg = linear_model.Lasso()

# Split the data into training (90%) and test sets (10%).
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=42
)

# `train` function is called `fit`, but mostly behaves the same as in OpenCV's `train`:
linreg.fit(X_train, y_train)

# -------------------------------------------------------------------------------------------------------------------

# mean squared error of our predictions: compare true housing
# prices of the training set (y_train) to predicted (linreg.predict(X_train)):
# print(metrics.mean_squared_error(y_train, linreg.predict(X_train)))

# The `score` method of the `linreg` object returns the coefficient of determination (R-squared) for training:
# print(linreg.score(X_train, y_train))

# -------------------------------------------------------------------------------------------------------------------
# ### Testing the model on unseen data
y_pred = linreg.predict(X_test)
# MSE gives us an idea of the `generalization performance` of the model:
# print(metrics.mean_squared_error(y_test, y_pred))

# The `score` method of the `linreg` object returns the coefficient of determination (R-squared):
# print(linreg.score(X_test, y_test))
# The same thing but works for all machine learning algorithms
# print(metrics.r2_score(y_test,y_pred))

# It isn't clear how good the model really is.
# Plot the data to help visualize the model performance:
plt.figure(figsize=(10, 6))
plt.plot(y_test, linewidth=3, label='ground truth')
plt.plot(y_pred, linewidth=3, label='predicted')
plt.legend(loc='best')
plt.xlabel('test data points')
plt.ylabel('target value')
# plt.show()

# Ground truth housing prices for all test samples in blue and predicted housing prices in red.
# What kind of prices causes the model to have increased error?

# Formalize the amount of variance in the data that we were able to explain by calculating R-squared:
plt.figure(figsize=(10, 6))
plt.plot(y_test, y_pred, 'o')
plt.plot([-10, 60], [-10, 60], 'k--')
plt.axis([-10, 60, -10, 60])
plt.xlabel('ground truth')
plt.ylabel('predicted')

scorestr = r'R$^2$ = %.3f' % linreg.score(X_test, y_test)
errstr = 'MSE = %.3f' % metrics.mean_squared_error(y_test, y_pred)
plt.text(-5, 50, scorestr, fontsize=12)
plt.text(-5, 45, errstr, fontsize=12)
plt.show()