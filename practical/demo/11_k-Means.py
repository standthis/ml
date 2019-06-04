#!/usr/bin/env python

# ---------------------------------------------------------------------------------------------------------------
# How k-Means works
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for better plot styling
import numpy as np

# 2D dataset containing four distinct blobs:
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50);
# plt.show()

# The k-means algorithm finds the specified number of clusters automatically:
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Visualize the results by plotting the data coloured by the labels (per cluster).
# Also show the cluster centres as determined by the k-means estimator:
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.show()

# Specify no. clusters for k-means and many other clustering algorithms like DBSCAN, mean-shift, or affinity propagation
# Gaussian Mixture Models is a complicated approach that finds quantitative measure of the fitness per number of clusters
# Ask the algorithm to identify six clusters, it will happily proceed and find the best six clusters:
labels = KMeans(6, random_state=42).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#             s=50, cmap='viridis');
# plt.show()


# ---------------------------------------------------------------------------------------------------------------
# k-Means is limited to linear cluster boundaries
# Points will be closer to their own cluster center than to others so k-Means fails for complicated geometries:
from sklearn.datasets import make_moons
X, y = make_moons(120, noise=.05, random_state=42)

labels = KMeans(2, random_state=42).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#             s=50, cmap='viridis')
# plt.show()

# Kernelized k-means is implemented in Scikit-Learn within the SpectralClustering estimator.
# It uses the graph of nearest neighbors to compute a higher-dimensional representation of the data:
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,
#             s=50, cmap='viridis')
# plt.show()


# ---------------------------------------------------------------------------------------------------------------
# k-means on digits
from sklearn.datasets import load_digits
digits = load_digits()

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)

# The result is 10 clusters in 64 dimensions:
# print(kmeans.cluster_centers_.shape)

# Visualize the cluster centers per digit:
# fig, ax = plt.subplots(2, 5, figsize=(8, 3))
# centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
# plt.show()

 # Fix digits 1 and 8 by matching each learned cluster label with the true labels found in them:
from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# With the ground truth labels measure accuracy of unsupervised clustering of digits:
from sklearn.metrics import accuracy_score
# print(accuracy_score(digits.target, labels))

# Check the confusion matrix to see when it misclassified:
# from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(digits.target, labels)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=digits.target_names,
#             yticklabels=digits.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label');
# plt.show()


# Improve the accuracy with preprocessing using t-distributed stochastic neighbor embedding (t-SNE) algorithm
# t-SNE is a nonlinear embedding algorithm that is particularly adept at preserving points within clusters:
from sklearn.manifold import TSNE

# Project the data:
tsne = TSNE(n_components=2, init='random', random_state=42)
digits_proj = tsne.fit_transform(digits.data)

# Compute the clusters
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(digits_proj)

# Permute the labels
labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0]

# Compute the accuracy
# print(accuracy_score(digits.target, labels))


# ---------------------------------------------------------------------------------------------------------------
# k-Means for color compression

# Note: this requires pillow
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(china)
# plt.show()

data = china / 255.0 # use 0...1 scale
data = data.reshape(427 * 640, 3)

# Visualize these pixels in this color space, using a subset of 10,000 pixels for efficiency:
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20)

plot_pixels(data, title='Input color space: 16 million possible colors')
# plt.show()

# Reduce 16 million colors to just 16 colors, using a k-Means clustering across the pixel space.
# Use mini batch k-Means to speed up:
import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors,
            title="Reduced color space: 16 colors")


# The result is a re-coloring of the original pixels, where each pixel is assigned the color of its closest cluster centre:
china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)
# plt.show()
