#!/usr/bin/env python

# -------------------------------------------------------------------------------------------------------------------
# Implementing k-NN in OpenCV
# -------------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

plt.style.use('ggplot')

# -------------------------------------------------------------------------------------------------------------------
#Simple example for generating training data
np.random.seed(42) #again for reproducibilty

# Generate a coordinate such that `0 <= x <= 100` and `0 <= y <= 100`:
single_data_point = np.random.randint(0, 100, 2)
print(single_data_point)

# Assign a label to the data point:
single_label = np.random.randint(0, 2)
print(single_label)

#End of simple example
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# Create a function instead that allows us to specify the amount of data and features
# num_samples is the number of data points
# num_features is the number of features every data point has
# note that class labels are randomly assigned to random data points
# note every row is a data point and one label per row
def generate_data(num_samples, num_features):
    """Randomly generates a number of data points"""
    data_size = (num_samples, num_features)
    data = np.random.randint(0, 100, size=data_size) # N X 2 array of random integers [0, 100]
    labels_size = (num_samples, 1)
    labels = np.random.randint(0, 2, size=labels_size)
    
    return data.astype(np.float32), labels
# -------------------------------------------------------------------------------------------------------------------

# Test the generate_data function with an 11 x 2 array. Note the dual tuple return:
train_data, train_labels = generate_data(11,2)
# print(train_data)
# print(train_labels)

# Inspect the first data point and corresponding label:
print(train_data[0], train_labels[0])

# The first data point (71, 60) is a blue square ('sb') and belongs to class 0

test = plt.plot(train_data[0, 0], train_data[0, 1], 'sb')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
# plt.savefig("First class")
# plt.show()

# -------------------------------------------------------------------------------------------------------------------
# Better yet, create a function that visualizes the whole training set.
# Input a list of all the data points that are blue squares and a second list for red triangles
def plot_data(all_blue, all_red):
    plt.figure(figsize=(10, 6))
    plt.scatter(all_blue[:, 0], all_blue[:, 1], c='b', marker='s', s=180)
    plt.scatter(all_red[:, 0], all_red[:, 1], c='r', marker='^', s=180)
    plt.xlabel('x coordinate (feature 1)')
    plt.ylabel('y coordinate (feature 2)')
# -------------------------------------------------------------------------------------------------------------------


# Next, split all the data points into red and blue sets.
# ravel flattens the array to easily select all the elements of the `train_labels` array created earlier that are equal to 0
# print("Original array",train_labels)
# print("Flattened array (1D)",train_labels.ravel())
# print("True when label = 1",train_labels.ravel() == 1)

# Blue data points are rows in `train_data` and set to label 0:
blue = train_data[train_labels.ravel() == 0]

# Red data points are rows in `train_data` and set to label 1:
red = train_data[train_labels.ravel() == 1]

# Finally, call the custom plot function:
plot_data(blue, red)
# plt.show()

# -------------------------------------------------------------------------------------------------------------------
# Train the classifier by following the "cvML Methodology" outline in the slides
# Use OpenCV's `ml` module:
knnModel = cv2.ml.KNearest_create()

# Pass training data to the train method:
knnModel.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# cv2.ml.ROW_SAMPLE tells k-NN the data is an N X 2 array i.e. every row is a data point
# train function returns True if data is the correct format
# -------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------
# Predict the label of a new data point
# `findNearest` predicts the label of a new data point based on its k-nearest neighbours.
# Use our `generate_data` function to generate a new (unseen) data point!
# Think of this new data point as a dataset of size 1:
newcomer, _ = generate_data(1,2) #Test data does not have a label thus the "ignore" underscore variable
# The idea of the previous line is for the classifier to predict the (unseen) label
# print(newcomer)

# Plot the training set once again, but add the new data point (from test set) as a green circle (since it is unseen
# data):
plot_data(blue, red)
plt.plot(newcomer[0, 0], newcomer[0, 1], 'go', markersize=14)
# plt.show()
# Forget the classifier, visually what would you guess the label is based on its neighbours, blue or red?
# Depends, doesn't it?
# NN is(86, 73)

# so what our classifier would predict for k=1:
ret, results, neighbour, dist = knnModel.findNearest(newcomer, 1)
# print("Predicted label:\t", results)
# print("neighbour's label:\t", neighbour)
# print("Distance to neighbour:\t", dist)

# The same would be true if we looked at the k=2 nearest neighbours (and k=3)
# Do not to pick even numbers for k. Why? Refer to the slides or stay tuned

# What would happen if k=7?
ret, results, neighbour, dist = knnModel.findNearest(newcomer, 7)
# print("Predicted label:\t", results)
# print("neighbour's label:\t", neighbour)
# print("Distance to neighbour:\t", dist)

# Clearly, k-NN uses majority voting since four blue squares (label 0), and only three red triangles (label 1)

# What would happen if k=6?
ret, results, neighbours, dist = knnModel.findNearest(newcomer, 6)
# print("Predicted label:\t", results)
# print("neighbours' labels:\t", neighbours)
# print("Distance to neighbours:\t", dist)


# Following the cvML Methodology the `predict` method can instead be used. But first set `k`:
knnModel.setDefaultK(7)
ret, pred = knnModel.predict(newcomer)
# print(pred)
knnModel.setDefaultK(6)
ret, pred = knnModel.predict(newcomer)
# print(int(pred))
knnModel.setDefaultK(1)
ret, pred = knnModel.predict(newcomer)
# print(int(pred))

# Which value for $k$ is the most suitable? A naive solution is just to try a bunch of values for k
# Better solutions will be covered later

# generate test data. This time we have 10 data points to predict:
test_data, test_labels = generate_data(11,2)
# print(test_data.ravel())
# print(test_labels.ravel())

knnModel.setDefaultK(3)
ret, pred = knnModel.predict(test_data)
opencv_pred = pred.flatten().astype(int)
# print(opencv_pred)

#According the cvML Methodology:
# print("Generic accuracy_score: {:.2f}%".format(accuracy_score(test_labels.ravel(), opencv_pred) *100))
# print("Generic precision_score: {:.2f}%".format(precision_score(test_labels.ravel(), opencv_pred) *100))
# print("Generic recall_score: {:.2f}%".format(recall_score(test_labels.ravel(), opencv_pred) *100))
# Note that these are scikit learn metric functions that work for OpenCV with the help of ravel()
# # -------------------------------------------------------------------------------------------------------------------

# The Scikit way
# -------------------------------------------------------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_data, train_labels.ravel())
acc = model.score(test_data, test_labels)
# print("k-NN's version of accuracy_score: {:.2f}%".format(acc * 100))
scikit_pred = model.predict(test_data)
# notice how Scikit stores predictions as a single item separated by space unlike OpenCV
# print(scikit_pred)

#According the cvML Methodology:
print("Generic accuracy_score: {:.2f}%".format(accuracy_score(test_labels.ravel(), scikit_pred) *100))
print("Generic precision_score: {:.2f}%".format(precision_score(test_labels.ravel(), scikit_pred) *100))
print("Generic recall_score: {:.2f}%".format(recall_score(test_labels.ravel(), scikit_pred) *100))

# -------------------------------------------------------------------------------------------------------------------
