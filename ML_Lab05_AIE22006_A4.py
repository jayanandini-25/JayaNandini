import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate training data
np.random.seed(42)  # Set seed for reproducibility
num_points = 20
X_train = np.random.randint(1, 11, size=(num_points, 2))  # Generate random values for features X and Y
classes_train = np.random.randint(0, 2, size=num_points)  # Assign random classes (0 or 1) to training data

# Create kNN classifier with k=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Fit kNN classifier on training data
knn_classifier.fit(X_train, classes_train)

# Generate test set data
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_values, y_values)
test_data = np.c_[xx.ravel(), yy.ravel()]

# Classify test points using kNN classifier
predicted_labels = knn_classifier.predict(test_data)

# Separate test points based on their predicted class labels
class0_test_points = test_data[predicted_labels == 0]
class1_test_points = test_data[predicted_labels == 1]

# Plot the scatter plot of the test data output
plt.figure(figsize=(8, 6))
plt.scatter(class0_test_points[:, 0], class0_test_points[:, 1], color='blue', alpha=0.5, label='Predicted class0')
plt.scatter(class1_test_points[:, 0], class1_test_points[:, 1], color='red', alpha=0.5, label='Predicted class1')
plt.title('Scatter Plot of Test Data Output')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
