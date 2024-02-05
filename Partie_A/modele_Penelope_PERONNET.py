# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:07:46 2024

@author: Admin
"""

# we are oing to do a knn predictive modele 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load Iris dataset from CSV
iris_df = pd.read_csv('iris.csv')  

# Reduce the size of the dataset
sample_size = 25 
iris_df_sample = iris_df.sample(n=sample_size, random_state=42)

# Separate features and target variable
X = iris_df_sample[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris_df_sample['variety']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create KNN classifier
k = 3  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
