#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(4242)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sys import exit
import sys

# -------------------------------------------------------------------------------------------------------------------
# Exercise 1
n_samples = 500
n_features = 2
X1 = np.random.rand(n_samples, n_features)
y1 = np.ones((n_samples, 1))
idx_neg = (X1[:, 0] - 0.5) ** 2 + (X1[:, 1] - 0.5) ** 2 < 0.03
y1[idx_neg] = 0

y1.ravel()
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X1[:, 0],-1), np.reshape(X1[:, 1],-1), c=np.reshape(y1,-1),s=100)
# plt.show()
# Code solution 1 here:
#pca = PCA(n_components=120, whiten=True, random_state=42)
#svc = SVC(kernel='rbf', class_weight='balanced', gamma='scale')
#model = make_pipeline(pca, svc)
svc = SVC(kernel='rbf', class_weight='balanced', gamma='scale')
#svc = SVC(kernel='linear')
#svc = SVC(kernel='poly')
#svc = SVC(kernel='sigmoid')
X_train, X_test, y_train, y_test = train_test_split(X1, y1,random_state=42) # using default test_size 
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Ex 1 -> Best rbf")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# ADJUST C GRID
#tune_param = {'svc__C': [1, 5, 10, 50],
#              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
#grid = GridSearchCV(model, tune_param, cv=3)
#
#grid.fit(X_train, y_train)
#print(grid.best_params_)

# -------------------------------------------------------------------------------------------------------------------
# Exercise 2
X2 = np.random.rand(n_samples, n_features)
y2 = np.ones((n_samples, 1))
idx_neg = (X2[:, 0] < 0.5) * (X2[:, 1] < 0.5) + (X2[:, 0] > 0.5) * (X2[:, 1] > 0.5)
y2[idx_neg] = 0

y2 = y2.ravel()
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X2[:, 0],-1), np.reshape(X2[:, 1],-1), c=np.reshape(y2,-1),s=100)
# plt.show()
# Code solution 2 here:
svc = SVC(kernel='rbf', class_weight='balanced', gamma='scale') # BEST 
#svc = SVC(kernel='linear')
#svc = SVC(kernel='poly')
#svc = SVC(kernel='sigmoid')
X_train, X_test, y_train, y_test = train_test_split(X2, y2,random_state=42) # using default test_size 
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Ex 2 -> best rbf")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# -------------------------------------------------------------------------------------------------------------------
# Exercise 3 - > FIX  ARRAYS BROKEN
rho_pos = np.random.rand(n_samples // 2, 1) / 2.0 + 0.5
rho_neg = np.random.rand(n_samples // 2, 1) / 4.0
rho = np.vstack((rho_pos, rho_neg))
phi_pos = np.pi * 0.75 + np.random.rand(n_samples // 2, 1) * np.pi * 0.5
phi_neg = np.random.rand(n_samples // 2, 1) * 2 * np.pi
phi = np.vstack((phi_pos, phi_neg))
X3 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
y3 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X3[:, 0],-1), np.reshape(X3[:, 1],-1), c=np.reshape(y3,-1),s=100)
# plt.show()


# Code solution 3 here:
X3 = X3.reshape(500, 2)
y3 = y3.ravel()
#svc = SVC(kernel='rbf', class_weight='balanced', gamma='scale') # BEST 
svc = SVC(kernel='linear')
#svc = SVC(kernel='poly')
#svc = SVC(kernel='sigmoid')
X_train, X_test, y_train, y_test = train_test_split(X3, y3,random_state=42) # using default test_size 
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Ex 3 -> best linear")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# -------------------------------------------------------------------------------------------------------------------
# Exercise 4
rho_pos = np.linspace(0, 2, n_samples // 2)
rho_neg = np.linspace(0, 2, n_samples // 2) + 0.5
rho = np.vstack((rho_pos, rho_neg))
phi_pos = 2 * np.pi * rho_pos
phi = np.vstack((phi_pos, phi_pos))
X4 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
y4 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X4[:, 0],-1), np.reshape(X4[:, 1],-1), c=np.reshape(y4,-1),s=100)
#X4 = X4[0][0]
#plt.show()

X4 = X4.ravel()
X4 = np.array(np.split(X4, 500))
y4 = y4.ravel()
# Code solution 4 here:
#svc = SVC(kernel='rbf', class_weight='balanced', gamma='auto')
svc = SVC(kernel='rbf', class_weight='balanced', gamma=1, C=1e4)
#svc = SVC(kernel='linear')
#svc = SVC(kernel='poly')
#svc = SVC(kernel='sigmoid')
X_train, X_test, y_train, y_test = train_test_split(X4, y4,random_state=42) # using default test_size 
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Ex 4 -> best rbf")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
