# KNN-Algorithm
Used Python, scikit-learn, pandas, numpy, and matplotlib to implement K-Nearest Neighbors (KNN). A non-parametric algorithm for versatile classification and regression, KNN predicts by leveraging nearest neighbors. 

# K-Nearest Neighbors (KNN) Implementation

This repository contains a Python implementation of the K-Nearest Neighbors (KNN) algorithm using scikit-learn.

# The Algorithm for KNN
from sklearn.neighbors import KNeighborsClassifier

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn.predict(X_test)

