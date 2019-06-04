#!/usr/bin/env python

# ---------------------------------------------------------------------------------------------------------------
# Training a shallow vs deep (Convolutional Layer) Neural Net to Classify Handwritten Digits Using Keras

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
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
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255
X_test = X_test / 255

# One-hot encode the training labels.
# Transforms the vector of class integers into a binary matrix:
num_classes = 10
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# The input layer expects images with the structure outline above [pixels][width][height].
# The first hidden layer is a convolutional layer called a Convolution2D with 32 5×5 feature maps and relu activation.
# Pooling layer that takes the max called MaxPooling2D. It is configured with a pool size of 2×2.
# The next layer is a regularization layer using Dropout.
# It is configured to randomly exclude 20% of neurons in the layer in order to reduce overfitting.
# Next convert the 2D matrix data to a Flattened vector for the output to be process fully connected layers.
# Another fully connected layer with 128 neurons and relu
# Finally, Softmax activation on output layer: probability values converted to 1 of 10 as the output prediction.

# define baseline model as above:
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	# Again using Logarithmic loss function and ADAM gradient descent algorithm as it is quick to learn the weights.

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the larger model, but not for potato pc. Beware of overfitting and bad parameters.
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
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
print("CNN Error: %.2f%%" % (100-scores[1]*100))