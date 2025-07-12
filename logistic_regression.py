import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class LogisticRegressionFromScratch:
    """
    Logistic Regression implementation from scratch using gradient descent
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost function"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Parameters:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)
            
            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[i-1] - cost) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def predict(self, X):
        """Make predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

def generate_sample_data(n_samples=1000, n_features=2, noise=0.1):
    """Generate sample data for logistic regression"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create a linear combination with some noise
    linear_combination = 0.5 * X[:, 0] + 0.3 * X[:, 1] + noise * np.random.randn(n_samples)
    
    # Convert to binary labels using sigmoid
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = (probabilities > 0.5).astype(int)
    
    return X, y

def demonstrate_sklearn_logistic_regression():
    """Demonstrate logistic regression using scikit-learn"""
    print("=== Scikit-learn Logistic Regression Demo ===\n")
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=1000, n_features=2)
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    df['Target'] = y
    
    print("Sample Data:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Target'].value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model coefficients: {model.coef_[0]}")
    print(f"Model intercept: {model.intercept_[0]}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, scaler, X_test, y_test, y_pred

def demonstrate_custom_logistic_regression():
    """Demonstrate logistic regression using custom implementation"""
    print("\n=== Custom Logistic Regression Demo ===\n")
    
    # Generate sample data
    X, y = generate_sample_data(n_samples=1000, n_features=2)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the custom model
    custom_model = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
    custom_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_custom = custom_model.predict(X_test_scaled)
    y_pred_proba_custom = custom_model.predict_proba(X_test_scaled)
    
    # Evaluate the model
    accuracy_custom = accuracy_score(y_test, y_pred_custom)
    print(f"Custom Model Accuracy: {accuracy_custom:.4f}")
    print(f"Custom Model Weights: {custom_model.weights}")
    print(f"Custom Model Bias: {custom_model.bias}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_custom))
    
    # Plot cost history
    custom_model.plot_cost_history()
    
    return custom_model, scaler, X_test, y_test, y_pred_custom

def create_comprehensive_example():
    """Create a comprehensive example with real-world-like data"""
    print("\n=== Comprehensive Example: Customer Purchase Prediction ===\n")
    
    # Generate more realistic data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: Age, Income, Time_on_site, Previous_purchases
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    time_on_site = np.random.exponential(5, n_samples)
    previous_purchases = np.random.poisson(2, n_samples)
    
    # Create feature matrix
    X = np.column_stack([age, income, time_on_site, previous_purchases])
    
    # Create target based on realistic relationships
    # Higher income, more time on site, and more previous purchases increase purchase probability
    linear_combination = (
        0.02 * (age - 35) +
        0.00001 * (income - 50000) +
        0.3 * time_on_site +
        0.5 * previous_purchases +
        np.random.normal(0, 1, n_samples)
    )
    
    # Convert to probabilities and binary labels
    probabilities = 1 / (1 + np.exp(-linear_combination))
    y = (probabilities > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Time_on_Site': time_on_site,
        'Previous_Purchases': previous_purchases,
        'Will_Purchase': y
    })
    
    print("Customer Data Sample:")
    print(df.head())
    print(f"\nPurchase rate: {y.mean():.2%}")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Model Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Feature importance
    feature_names = ['Age', 'Income', 'Time_on_Site', 'Previous_Purchases']
    coefficients = model.coef_[0]
    
    print("\nFeature Importance (Coefficients):")
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")
    
    return model, df, scaler

def main():
    """Main function to run all demonstrations"""
    print("🚀 Logistic Regression Implementation in Python")
    print("=" * 50)
    
    # Demonstrate sklearn implementation
    sklearn_model, sklearn_scaler, X_test, y_test, y_pred = demonstrate_sklearn_logistic_regression()
    
    # Demonstrate custom implementation
    custom_model, custom_scaler, X_test_custom, y_test_custom, y_pred_custom = demonstrate_custom_logistic_regression()
    
    # Comprehensive example
    comprehensive_model, customer_df, comprehensive_scaler = create_comprehensive_example()
    
    print("\n🎉 All demonstrations completed successfully!")
    print("\nKey takeaways:")
    print("1. Logistic regression is great for binary classification")
    print("2. Feature scaling is important for convergence")
    print("3. The sigmoid function maps any real number to (0,1)")
    print("4. Cross-validation helps prevent overfitting")
    print("5. Both sklearn and custom implementations can work well")

if __name__ == "__main__":
    main()