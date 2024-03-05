import numpy as np
import matplotlib.pyplot as plt

#A3: Repeat exercise A1 with varying the learning rate, keeping the initial weights same. Take learning rate = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}. Make a plot of the number of iterations taken forlearning to converge against the learning rates.

# Define the AND gate input-output pairs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Initial weights
W = np.array([10, 0.2, -0.75])

# Define the perceptron function
def perceptron(x, w):
    return 1 if np.dot(x, w) >= 0 else 0

# Define the training function
def train_perceptron(learning_rate):
    global W
    epochs = 0
    max_epochs = 1000
    convergence_error = 0.002
    
    while True:
        error = 0
        for i in range(len(X)):
            y_pred = perceptron(np.insert(X[i], 0, 1), W)
            delta = Y[i] - y_pred
            W += learning_rate * delta * np.insert(X[i], 0, 1)
            error += delta ** 2
        epochs += 1
        if error <= convergence_error or epochs >= max_epochs:
            break
            
    return epochs

# Learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Number of iterations for each learning rate
iterations = []

# Train for each learning rate
for lr in learning_rates:
    W = np.array([10, 0.2, -0.75])  
    epochs = train_perceptron(lr)
    iterations.append(epochs)

# Plotting
plt.plot(learning_rates, iterations, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations')
plt.title('Number of Iterations vs Learning Rate')
plt.grid(True)
plt.show()
