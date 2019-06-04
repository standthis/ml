# -------------------------------------------------------------------------------------------------------------------
# Scaling Data (or Standardization) refers to the process of scaling the data to have zero mean and 1 unit variance.
# Scikit-learn offers a preprocessing module that includes scaling.

from sklearn import preprocessing

import numpy as np

# Consider a 3 x 3 data matrix `X`, standing for three data points (rows) with three arbitrarily
# chosen feature values each (columns):
X = np.array([[ 1., -2.,  2.],
              [ 3.,  0.,  0.],
              [ 0.,  1., -1.]])

# -------------------------------------------------------------------------------------------------------------------
# `scale` standardizes the data matrix `X`:
X_scaled = preprocessing.scale(X)
# print(X_scaled)

# Check that `X_scaled` is scaled correctly with (close to) zero mean
# print(X_scaled.mean(axis=0))

# Check that every row of the scaled feature matrix has variance of 1 (unit variance):
# print(X_scaled.std(axis=0))

# -------------------------------------------------------------------------------------------------------------------
# Normalization is the process of scaling individual samples to have unit norm (length of a vector).
# L1 norm is the Manhattan distance and L2 norm is the well-known Euclidean distance

# `X` can be normalized to the l1 or l2 norm by specifying the `norm` keyword, distances are higher for l2:
X_normalized_l1 = preprocessing.normalize(X, norm='l1')
# print(X_normalized_l1)
# L2 norm
X_normalized_l2 = preprocessing.normalize(X, norm='l2')
# print(X_normalized_l2)

# -------------------------------------------------------------------------------------------------------------------
# ## Scaling features to a range
# Often the range is zero and one, so that the maximum absolute value of each feature is scaled to unit size.
# Use `MinMaxScaler()` without parameters to achieve this
min_max_scaler = preprocessing.MinMaxScaler()
X_min_max = min_max_scaler.fit_transform(X)
# print(X_min_max)
# Or passing a keyword argument `feature_range` to the `MinMaxScaler` constructor:
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-10, 10))
X_min_max2 = min_max_scaler.fit_transform(X)
# print(X_min_max2)

# -------------------------------------------------------------------------------------------------------------------
# Binarizing features
# Check whether a feature is present or absent.
# Assume that numbers between 0 and 1 represent amounts between 0 and one billion dollars in an account. More than
# 0.5 billion dollars in the account, means that person is rich, which is represented with a 1. Else we put a 0.

# Threshold the data with `threshold=0.5`:
binarizer = preprocessing.Binarizer(threshold=0.5)
X_binarized = binarizer.transform(X)
# print(X_binarized)

# Handling missing data automatically:
from numpy import nan
X = np.array([[ nan, 0,   3  ],
              [ 2,   9,  -8  ],
              [ 1,   nan, 1  ],
              [ 5,   2,   4  ],
              [ 7,   6,  -3  ]])

# Obvious solution is to replace all the nan values with some appropriate fill values, known as imputation
# Scikit-learn supports three different impute methods:
# - `'mean'`: Replaces all nan values with mean value of specified axis of the matrix (default: axis=0)
# - `'median'`: Replaces all nan values with median value of specified axis of the matrix (default: axis=0)
# - `'most_frequent'`: Replaces all nan values with most frequent value of specified axis of the matrix (default: axis=0)

# `'mean'` imputer can be called as follows:
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer #This is deprecated
imp = SimpleImputer(strategy='mean')
X2 = imp.fit_transform(X)
# print(X2)

# Verify the maths:
# print(np.mean(X[1:, 0]), X2[0, 0])

# Similarly, `'median'` imputer:
imp = SimpleImputer(strategy='median')
X3 = imp.fit_transform(X)
# print(X3)

# Verify the maths:
# print(np.median(X[1:, 0]), X3[0, 0])