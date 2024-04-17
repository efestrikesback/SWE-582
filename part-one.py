import numpy as np

# 1.1 Implementing PLA
 
# Load the datasets
data_large = np.load('data_large.npy')
label_large = np.load('label_large.npy')
data_small = np.load('data_small.npy')
label_small = np.load('label_small.npy')

def pla(data, labels, max_iter=1000000):
    # Randomly initializing weights
    weights = np.random.rand(data.shape[1])
    iteration = 0
    all_correct = False

    while not all_correct and iteration < max_iter:
        all_correct = True
        for i in range(data.shape[0]):
            if np.sign(np.dot(data[i], weights)) != labels[i]:
                # Update rule: w = w + y*x
                weights += labels[i] * data[i]
                all_correct = False  # If any update is done, not all are correctly classified
        iteration += 1
    
    return weights, iteration

# Train PLA on both datasets
weights_large, iterations_large = pla(data_large, label_large)
weights_small, iterations_small = pla(data_small, label_small)

print(f"iterations_large: {iterations_large}")
print(f"iterations_small: {iterations_small}")

# 1.2 Plotting
import matplotlib.pyplot as plt

def plot_data(data, labels, weights, title):

    pos = data[labels == 1]
    neg = data[labels == -1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pos[:, 1], pos[:, 2], color='blue', marker='o', label='Class +1')
    plt.scatter(neg[:, 1], neg[:, 2], color='red', marker='x', label='Class -1')
    
    # Add decision boundary
    x_values = np.array([data[:, 1].min(), data[:, 1].max()])
    y_values = -(weights[1] * x_values + weights[0]) / weights[2]
    plt.plot(x_values, y_values, 'k-')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for the small dataset
plot_data(data_small, label_small, weights_small, "PLA Decision Boundary Small Dataset")

# Plot for the large dataset
plot_data(data_large, label_large, weights_large, "PLA Decision Boundary Large Dataset")

# 1.3 Sensitivity to Initialization

# Repeat PLA training with random initial weights
def multiple_initializations(data, labels, trials=10):
    weights_list = []
    iterations_list = []
    
    for _ in range(trials):
        weights, iterations = pla(data, labels)
        weights_list.append(weights)
        iterations_list.append(iterations)
    
    return weights_list, iterations_list

# Perform the experiment with the small dataset
weights_list_small, iterations_list_small = multiple_initializations(data_small, label_small)

# The iterations needed and weights for each trial
iterations_list_small, weights_list_small

import pandas as pd

weights_df = pd.DataFrame({
    "Trial": range(1, 11),
    "Iterations": iterations_list_small,
    "Weight 0": [weights[0] for weights in weights_list_small],
    "Weight 1": [weights[1] for weights in weights_list_small],
    "Weight 2": [weights[2] for weights in weights_list_small]
})

print(weights_df)
