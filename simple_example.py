#!/usr/bin/env python3
"""
Simple example demonstrating how to use the logistic regression implementations
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from logistic_regression import LogisticRegressionFromScratch

def main():
    print("Simple Logistic Regression Example")
    print("=" * 40)
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Method 1: Using scikit-learn
    print("\n1. Using scikit-learn:")
    sklearn_model = LogisticRegression(random_state=42)
    sklearn_model.fit(X_train_scaled, y_train)
    sklearn_accuracy = sklearn_model.score(X_test_scaled, y_test)
    print(f"   Accuracy: {sklearn_accuracy:.4f}")
    
    # Method 2: Using custom implementation
    print("\n2. Using custom implementation:")
    custom_model = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
    custom_model.fit(X_train_scaled, y_train)
    custom_predictions = custom_model.predict(X_test_scaled)
    custom_accuracy = np.mean(custom_predictions == y_test)
    print(f"   Accuracy: {custom_accuracy:.4f}")
    
    # Make a prediction on new data
    print("\n3. Making predictions on new data:")
    new_data = np.array([[0.5, -0.3], [-0.2, 1.1]])
    new_data_scaled = scaler.transform(new_data)
    
    sklearn_pred = sklearn_model.predict(new_data_scaled)
    custom_pred = custom_model.predict(new_data_scaled)
    
    print(f"   New data points: {new_data}")
    print(f"   Sklearn predictions: {sklearn_pred}")
    print(f"   Custom predictions: {custom_pred}")
    
    # Get prediction probabilities
    sklearn_proba = sklearn_model.predict_proba(new_data_scaled)
    custom_proba = custom_model.predict_proba(new_data_scaled)
    
    print(f"   Sklearn probabilities: {sklearn_proba[:, 1]}")
    print(f"   Custom probabilities: {custom_proba}")

if __name__ == "__main__":
    main()