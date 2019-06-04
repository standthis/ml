#!/usr/bin/env python

# ---------------------------------------------------------------------------------------------------------------
# # Implementing a Multi-Layer Perceptron (MLP) in OpenCV
from sklearn.datasets.samples_generator import make_blobs
X_raw, y_raw = make_blobs(n_samples=100, centers=2,
                          cluster_std=5.2, random_state=42)

# Preprocessing the data
# OpenCV works with 32-bit floating point numbers:
import numpy as np
X = X_raw.astype(np.float32)

# Perform one-hot encoding.
# This will make sure each category of target labels can be assigned to a neuron in the output layer:
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False, dtype=np.float32)
y = enc.fit_transform(y_raw.reshape(-1, 1))

# Create the OpenCV model
import cv2
mlp = cv2.ml.ANN_MLP_create()

# Since the data matrix X has two features, the first layer should also have two neurons in it
# Since the output has two different values (binary), the last layer should also have two neurons in it
# Hidden layers are more arbitrary. Try 8:
n_input = 2
n_hidden = 8
n_output = 2
mlp.setLayerSizes(np.array([n_input, n_hidden, n_output]))

# - `mlp.setActivationFunction`: This defines the activation function to be used for every neuron in the network
# - `mlp.setTrainMethod`: This defines a suitable training method. i.e. Backpropagation
# - `mlp.setTermCriteria`: This sets the termination criteria of the training phase
# 
# OpenCV provides three activation functions as options:
# - `cv2.ml.ANN_MLP_IDENTITY`: This is the linear activation function
# - `cv2.ml.ANN_MLP_SIGMOID_SYM`: Symmetrical sigmoid function known as hyperbolic tangent (tanh)
# - `cv2.ml.ANN_GAUSSIAN`: This is the Gaussian function (bell curve)
# Use a sigmoid function that squashes the input values into the range [0, 1].
# alpha controls the slope of the function, beta defines the upper and lower bounds of the output.
alpha = 2.5
beta = 1.0

mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, alpha, beta)

# Activation function looks like:
import matplotlib.pyplot as plt
plt.style.use('ggplot')


x_sig = np.linspace(-1.0, 1.0, 100)
y_sig = beta * (1.0 - np.exp(-alpha * x_sig))
y_sig /= (1 + np.exp(-alpha * x_sig))
plt.figure(figsize=(10, 6))
plt.plot(x_sig, y_sig, linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Additional scaling factors can be set via `mlp.setBackpropMomentumScale` and `mlp.setBackpropWeightScale`.
# For now use the standard backpropagation learning algorithm:
mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

# Criteria for training to end when reaching max iterations or good accuracy (specified by epsilon):
term_mode = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
term_max_iter = 300
term_eps = 0.01
mlp.setTermCriteria((term_mode, term_max_iter, term_eps))

# Training and testing the MLP:
mlp.train(X, cv2.ml.ROW_SAMPLE, y)

# predicting target labels:
_, y_pred = mlp.predict(X)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred.round(), y))

# look at the new decision boundary:
def plot_decision_boundary(classifier, X_test, y_test):
    # create a mesh to plot in
    h = 0.02  # step size in mesh
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    X_hypo = np.c_[xx.ravel().astype(np.float32),
                   yy.ravel().astype(np.float32)]
    _, zz = classifier.predict(X_hypo)
    zz = np.argmax(zz, axis=1)
    zz = zz.reshape(xx.shape)
    
    plt.contourf(xx, yy, zz, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=200)

plt.figure(figsize=(10, 6))
plot_decision_boundary(mlp, X, y_raw)
plt.show()
