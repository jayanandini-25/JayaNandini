import numpy as np
import matplotlib.pyplot as plt

# XOR gate input-output pairs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

# Initial weights
W = np.array([10, 0.2, -0.75])

# Learning rate
alpha = 0.05

# Step activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron function
def perceptron(x, w):
    return step_function(np.dot(x, w))

# Calculate sum-square error
def calculate_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# Training the perceptron
error_values = []
epochs = 0
max_epochs = 1000
convergence_error = 0.002

while True:
    error = 0
    for i in range(len(X)):
        y_pred = perceptron(np.insert(X[i], 0, 1), W)
        delta = Y[i] - y_pred
        W += alpha * delta * np.insert(X[i], 0, 1)
        error += calculate_error(Y[i], y_pred)
    error_values.append(error)
    epochs += 1
    if error <= convergence_error or epochs >= max_epochs:
        break

# Plotting
plt.plot(range(1, epochs + 1), error_values)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs Error')
plt.grid(True)
plt.show()

print("Number of epochs needed for convergence:", epochs)
print("Learned weights:", W)
