import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#A1
class1_data = pd.read_csv('C:\\Users\\jayan\\Downloads\\B TECH\\Sem_4\\ML\\OneDrive_2024-02-20\\ML Lab\\Embedded Data\\DATA\\code_only.csv')
class2_data = pd.read_csv('C:\\Users\\jayan\\Downloads\\B TECH\\Sem_4\\ML\\OneDrive_2024-02-20\\ML Lab\\Embedded Data\\DATA\\code_comm.csv')

class1_vectors = class1_data['0'].values
class2_vectors = class2_data['0'].values

class1_vectors = np.array(class1_vectors)
class2_vectors = np.array(class2_vectors)

# Creating class labels
class1_labels = ['Class1'] * len(class1_vectors)
class2_labels = ['Class2'] * len(class2_vectors)

# Combine feature vectors and class labels
class1_df = pd.DataFrame({'Feature': class1_vectors, 'Class': class1_labels})
class2_df = pd.DataFrame({'Feature': class2_vectors, 'Class': class2_labels})

# Combine both classes into one DataFrame
data = pd.concat([class1_df, class2_df], ignore_index=True)

# Separate features (X) and labels (y)
X = data['Feature']
y = data['Class']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
neigh.fit(X_train.values.reshape(-1, 1), y_train) 

# Test the accuracy of the kNN classifier on the test set
accuracy = neigh.score(X_test.values.reshape(-1, 1), y_test)

print("Accuracy of the kNN classifier on the test set:", accuracy)

# Use the predict() function to predict classes for the test set
predicted_classes = neigh.predict(X_test.values.reshape(-1, 1))

# Confusion matrix for training data
train_predicted_classes = neigh.predict(X_train.values.reshape(-1, 1))
train_conf_matrix = confusion_matrix(y_train, train_predicted_classes)
print("Confusion Matrix for Training Data:")
print(train_conf_matrix)

# Precision, recall, and F1-score for training data
train_precision = precision_score(y_train, train_predicted_classes, average='weighted')
train_recall = recall_score(y_train, train_predicted_classes, average='weighted')
train_f1 = f1_score(y_train, train_predicted_classes, average='weighted')

print("Precision for Training Data:", train_precision)
print("Recall for Training Data:", train_recall)
print("F1-Score for Training Data:", train_f1)

# Confusion matrix for test data
conf_matrix = confusion_matrix(y_test, predicted_classes)
print("Confusion Matrix for Test Data:")
print(conf_matrix)

# Precision, recall, and F1-score for test data
precision = precision_score(y_test, predicted_classes, average='weighted')
recall = recall_score(y_test, predicted_classes, average='weighted')
f1 = f1_score(y_test, predicted_classes, average='weighted')

print("Precision for Test Data:", precision)
print("Recall for Test Data:", recall)
print("F1-Score for Test Data:", f1)


