# -------------------------------------------------------------------------------------------------------------------
# Reducing the Dimensionality of the Data
# Defeating the curse of dimensionality: for a given sample size, there exists an optimal number of features for best
# classification performance

# -------------------------------------------------------------------------------------------------------------------
# Implementing Principal Component Analysis (PCA) in OpenCV

# PCA rotates all data points they align with the two axes that explains the spread of the data. PCA aims to transform
# the data to a new coordinate system by means of an orthogonal linear transformation.
# Project the data onto the new coordinate system such that the first coordinate has the greatest variance.
# This is called the first principal component

# Here is some random data drawn from a multivariate Gaussian:
import numpy as np
from sklearn import decomposition

mean = [20, 20]
cov = [[5, 0], [25, 25]]
np.random.seed(42)
x, y = np.random.multivariate_normal(mean, cov, 1000).T

# Plot this data using Matplotlib. Look at the spread:
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', zorder=1)
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
# plt.show()

# Format the data by stacking the `x` and `y` coordinates as a feature matrix `X`:
X = np.vstack((x, y)).T

# Create an empty array `np.array([])` as a mask, telling OpenCV to use all data points in the feature matrix.
# Compute PCA on the feature matrix `X`
import cv2
mu, eig = cv2.PCACompute(X, np.array([]))
# print(eig)

# Note the following looks complicated but is simply for demonstration purposes (showing directional arrows of PC.
# Plot the eigenvectors of the decomposition on top of the data:
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', zorder=1)
plt.quiver(mean[0], mean[1], eig[:, 0], eig[:, 1], zorder=3, scale=0.2, units='xy')
plt.text(mean[0] + 5 * eig[0, 0], mean[1] + 5 * eig[0, 1], 'u1', zorder=5, 
         fontsize=16, bbox=dict(facecolor='white', alpha=0.6))
plt.text(mean[0] + 7 * eig[1, 0], mean[1] + 4 * eig[1, 1], 'u2', zorder=5, 
         fontsize=16, bbox=dict(facecolor='white', alpha=0.6))
plt.axis([0, 40, 0, 40])
plt.xlabel('feature 1')
plt.ylabel('feature 2')
# plt.show()

# PCA rotates the data so that the two axes (`x` and `y`) are aligned with the first two principal components.

# In OpenCV the PCAProject function both initiates and fits the transformed matrix to X2:
# X2 = cv2.PCAProject(X, mu, eig)

# Scikit: first instantiate using the `decomposition` module:
pca = decomposition.PCA(n_components=2)

# Now use the `fit_transform` method:
X2 = pca.fit_transform(X)

# The blob of data is rotated so that the most spread is along the `x` axis:
# Note the more even spread
plt.figure(figsize=(10, 6))
plt.plot(X2[:, 0], X2[:, 1], 'o')
plt.xlabel('first principal component')
plt.ylabel('second principal component')
plt.axis([-20, 20, -10, 10])
# plt.show()

# Implementing Independent Component Analysis (ICA) performs the same mathematical
# steps as PCA, but it chooses the components of the decomposition to be as independent as possible from each other.

# Again, first instantiate, then use the `fit_transform` method:
ica = decomposition.FastICA()
X2 = ica.fit_transform(X)

# Plot the projected data on the first two independent components:
plt.figure(figsize=(10, 6))
plt.plot(X2[:, 0], X2[:, 1], 'o')
plt.xlabel('first independent component')
plt.ylabel('second independent component')
plt.axis([-0.2, 0.2, -0.2, 0.2])
plt.show()
