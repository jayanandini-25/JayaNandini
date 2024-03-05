import numpy as np
import matplotlib.pyplot as plt

#A1: Develop the above perceptron in your own code (don’t use the perceptron model available from package). Use the initial weights as provided below. W0 = 10, W1 = 0.2, w2 = -0.75, learning rate (α) = 0.05. Use Step activation function to learn the weights of the network to implement above provided AND gate logic. The activation function is demonstrated below.Identify the number of epochs needed for the weights to converge in the learning process. Make a plot of the epochs against the error values calculated (after each epoch, calculate the sum-square error against all training samples).(Note: Learning is said to be converged if the error is less than or equal to 0.002. Stop the learning after 1000 iterations if the convergence error condition is not met.)

# AND gate input-output pairs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

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
