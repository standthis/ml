#!/usr/bin/env python

# ---------------------------------------------------------------------------------------------------------------
# Training a MLP to Classify Handwritten Digits Using Keras

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display 4 images as greyscale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
# plt.show()

# fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)

# Keras version of train_test_split on the mnist 28 x 28 = 784 dimensions
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# The neural nets in Keras act on the feature matrix slightly differently than the standard
# OpenCV and scikit-learn estimators.
# Reshape the feature matrix into a 4D matrix with dimensions n_features x 28 x 28 x 1:
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One-hot encode the training labels.
# Transforms the vector of class integers into a binary matrix:
num_classes = 10
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# MLP with one hidden layer and the same number of neurons as there are inputs (784).
# relu activation is used for the hidden layer.
# Softmax activation on output layer: probability values converted to 1 of 10 as the output prediction.

# define baseline model as above:
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	# Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras)
	# ADAM gradient descent algorithm is fast and is used to learn the weights.
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Fit and evaluate the model over 10 epochs with updates every 200 images (batches).
# The test data is used as the validation dataset, to see the improvement as the model as it trains.
# A verbose value of 2 is used to reduce the output to one line for each training epoch.

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))