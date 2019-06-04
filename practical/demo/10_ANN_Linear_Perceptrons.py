#!/usr/bin/env python

# ---------------------------------------------------------------------------------------------------------------
# To test our perceptron classifier, we need to create some mock data.
import numpy as np
from sklearn.linear_model import Perceptron

#  Generate 100 data samples, again relying on scikit-learn make_blobs
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=2,
                  cluster_std=2.2, random_state=42)

# Labels are either +1 or -1:
y = 2 * y - 1

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], s=100, c=y);
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()

# Fitting the perceptron to data
p = Perceptron(max_iter=10, tol=0.1)
p.fit(X, y)

# Learned bias and weights:
print(p.coef_, p.intercept_,"\n")

# ## Evaluating the perceptron classifier
from sklearn.metrics import accuracy_score
print(accuracy_score(p.predict(X), y))

def plot_decision_boundary(classifier, X_test, y_test):
    # create a mesh to plot in
    h = 0.02  # step size in mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    X_hypo = np.c_[xx.ravel().astype(np.float32),
                   yy.ravel().astype(np.float32)]
    zz = classifier.predict(X_hypo)
    zz = zz.reshape(xx.shape)
    
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=200)


plt.figure(figsize=(10, 6))
plot_decision_boundary(p, X, y)
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()

# Applying the perceptron to data that is not linearly separable enough:
X, y = make_blobs(n_samples=100, centers=2,
                  cluster_std=5.2, random_state=42)
y = 2 * y - 1

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], s=100, c=y);
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()

# Apply the perceptron classifier to this dataset:
p = Perceptron(max_iter=15, tol=0.1)
p.fit(X, y)

accuracy = accuracy_score(p.predict(X), y)
print('accuracy=',accuracy)

plt.figure(figsize=(10, 6))
plot_decision_boundary(p, X, y)
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()