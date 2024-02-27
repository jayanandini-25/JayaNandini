import numpy as np
import matplotlib.pyplot as plt

# Generate 20 data points with random values between 1 and 10 for features X and Y
np.random.seed(42)  # Set seed for reproducibility
num_points = 20
X = np.random.randint(1, 11, size=(num_points, 2))  # Generate random values for features X and Y

# Assign points to 2 different classes: class0 (Blue) and class1 (Red)
classes = np.random.randint(0, 2, size=num_points)

# Separate points based on their classes
class0_points = X[classes == 0]
class1_points = X[classes == 1]

# Plot the scatter plot of the training data
plt.figure(figsize=(8, 6))
plt.scatter(class0_points[:, 0], class0_points[:, 1], color='blue', label='class0')
plt.scatter(class1_points[:, 0], class1_points[:, 1], color='red', label='class1')
plt.title('Scatter Plot of Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
