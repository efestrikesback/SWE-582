import pandas as pd

# 2.1 Data Familiarization and Preparation

# Load the dataset
file_path = 'Rice_Cammeo_Osmancik.csv'
rice_data = pd.read_csv(file_path)

# Display the first few rows and basic information about the dataset 
rice_data.head(), rice_data.info(), rice_data.describe()

print(rice_data.head(), rice_data.info(), rice_data.describe())


import numpy as np

# Encode the 'Class' column
class_mapping = {'Cammeo': 0, 'Osmancik': 1}
rice_data['Class'] = rice_data['Class'].map(class_mapping)

# Normalize the numerical features
numerical_features = rice_data.columns[:-1]  # All columns except 'Class'
rice_data[numerical_features] = (rice_data[numerical_features] - rice_data[numerical_features].mean()) / rice_data[numerical_features].std()

# Split the data into train and test sets
np.random.seed(42)
mask = np.random.rand(len(rice_data)) < 0.8
train_data_manual = rice_data[mask]
test_data_manual = rice_data[~mask]

train_data_manual.head(), train_data_manual.shape, test_data_manual.shape
 
print(train_data_manual.head(), train_data_manual.shape, test_data_manual.shape)


# 2.2 5-fold cross-validation 

#Sigmoid activation function.
def sigmoid(z):
    
    return 1 / (1 + np.exp(-z))
#Prediction using logistic regression.
def logistic_regression_predict(weights, X):
    
    z = np.dot(X, weights)
    return sigmoid(z)

#Compute the cost for logistic regression with optional L2 regularization.
def logistic_regression_cost(weights, X, y, lambda_reg=0):

    m = len(y)
    predictions = logistic_regression_predict(weights, X)
    # Cost for logistic regression
    cost = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
    # Adding L2 regularization term if lambda_reg > 0
    if lambda_reg > 0:
        l2_cost = (lambda_reg / (2 * m)) * np.sum(weights ** 2)
        cost += l2_cost
    return cost

#Compute the gradients for logistic regression with optional L2 regularization.
def logistic_regression_gradient(weights, X, y, lambda_reg=0):

    m = len(y)
    predictions = logistic_regression_predict(weights, X)
    error = predictions - y
    gradient = np.dot(X.T, error) / m
    # Adding L2 regularization term if lambda_reg > 0
    if lambda_reg > 0:
        l2_gradient = (lambda_reg / m) * weights
        gradient += l2_gradient
    return gradient

#Perform full batch gradient descent for logistic regression.
def gradient_descent(X, y, learning_rate, iterations, lambda_reg=0):

    weights = np.zeros(X.shape[1])
    costs = []
    for i in range(iterations):
        gradient = logistic_regression_gradient(weights, X, y, lambda_reg)
        weights -= learning_rate * gradient
        cost = logistic_regression_cost(weights, X, y, lambda_reg)
        costs.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")
    return weights, costs

# Prepare feature matrix X and target vector y
X_train = train_data_manual.iloc[:, :-1].values  # all columns except the last one
y_train = train_data_manual.iloc[:, -1].values  # last column

# Parameters for the model
learning_rate = 0.01
iterations = 500

# Train logistic regression model using full batch gradient descent
weights_gd, costs_gd = gradient_descent(X_train, y_train, learning_rate, iterations)

weights_gd, costs_gd[-1]

print(weights_gd, costs_gd[-1])


# Perform stochastic gradient descent for logistic regression.
def stochastic_gradient_descent(X, y, learning_rate, iterations, batch_size=1, lambda_reg=0):

    weights = np.zeros(X.shape[1])
    costs = []
    for i in range(iterations):
        # Randomly shuffle the data indices
        indices = np.random.permutation(len(y))
        for idx in range(0, len(y), batch_size):
            batch_indices = indices[idx:idx+batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            gradient = logistic_regression_gradient(weights, X_batch, y_batch, lambda_reg)
            weights -= learning_rate * gradient
        
        # Calculate and store the cost after processing all batches
        cost = logistic_regression_cost(weights, X, y, lambda_reg)
        costs.append(cost)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")

    return weights, costs

# Train logistic regression model using stochastic gradient descent
weights_sgd, costs_sgd = stochastic_gradient_descent(X_train, y_train, learning_rate, iterations)

print(weights_sgd, costs_sgd[-1])



# Adjust the parameters for SGD to optimize computation time
# Reducing the number of iterations and increasing the batch size
iterations_sgd_optimized = 200  # Reduced number of iterations
batch_size_optimized = 10       # Increased batch size

# Train logistic regression model using optimized stochastic gradient descent
weights_sgd_optimized, costs_sgd_optimized = stochastic_gradient_descent(
    X_train, y_train, learning_rate, iterations_sgd_optimized, batch_size_optimized)

weights_sgd_optimized, costs_sgd_optimized[-1]



"""Perform k-fold cross-validation to find the optimal lambda for logistic regression with L2 regularization."""
def cross_validation(X, y, k_folds, lambda_values):
    
    fold_size = len(y) // k_folds
    validation_scores = {lam: [] for lam in lambda_values}
    
    for i in range(k_folds):
        start = i * fold_size
        end = start + fold_size
        if i == k_folds - 1:
            end = len(y)  # Ensure last fold includes the remainder

        X_valid = X[start:end]
        y_valid = y[start:end]
        X_train = np.concatenate([X[:start], X[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        
        for lam in lambda_values:
            weights, _ = gradient_descent(X_train, y_train, learning_rate, iterations, lambda_reg=lam)
            valid_cost = logistic_regression_cost(weights, X_valid, y_valid, lambda_reg=lam)
            validation_scores[lam].append(valid_cost)
    
    # Calculate average validation scores for each lambda
    average_scores = {lam: np.mean(scores) for lam, scores in validation_scores.items()}
    return average_scores

# Define a range of lambda values to test
lambda_values = [0.001, 0.01, 0.1, 1, 10]

# Perform 5-fold cross-validation to find the optimal lambda
average_scores = cross_validation(X_train, y_train, 5, lambda_values)
average_scores

"""Test resuslts have shown the optimal lambda as 0.001"""
"""Accucarcy comparison"""

# Calculate accuracy for logistic regression
def calculate_accuracy(weights, X, y):
    predictions = logistic_regression_predict(weights, X) >= 0.5
    return np.mean(predictions == y)

# Prepare the test set features and labels
X_test = test_data_manual.iloc[:, :-1].values
y_test = test_data_manual.iloc[:, -1].values

# Evaluate both models
# Non-Regularized Model
accuracy_train_nonreg = calculate_accuracy(weights_gd, X_train, y_train)
accuracy_test_nonreg = calculate_accuracy(weights_gd, X_test, y_test)

# Regularized Model (optimal_lambda obtained from cross-validation)
optimal_lambda = 0.01
weights_reg, _ = gradient_descent(X_train, y_train, learning_rate, iterations, lambda_reg=optimal_lambda)
accuracy_train_reg = calculate_accuracy(weights_reg, X_train, y_train)
accuracy_test_reg = calculate_accuracy(weights_reg, X_test, y_test)

# Output the results
print(f"Non-Regularized Model - Train Accuracy: {accuracy_train_nonreg}, Test Accuracy: {accuracy_test_nonreg}")
print(f"Regularized Model - Train Accuracy: {accuracy_train_reg}, Test Accuracy: {accuracy_test_reg}")
