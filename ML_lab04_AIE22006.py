import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class1_data = pd.read_csv('C:\\Users\\jayan\\Downloads\\code_only.csv')
class2_data = pd.read_csv('C:\\Users\\jayan\\Downloads\\code_comm.csv')
class3_data = pd.read_csv('C:\\Users\\jayan\\Downloads\\code_ques.csv')
class4_data = pd.read_csv('C:\\Users\\jayan\\Downloads\\code_sol.csv')

class1_vectors = class1_data['0'].values
class2_vectors = class2_data['0'].values
class3_vectors = class1_data['0'].values
class4_vectors = class2_data['0'].values

class1_vectors = np.array(class1_vectors)
class2_vectors = np.array(class2_vectors)
class3_vectors = np.array(class3_vectors)
class4_vectors = np.array(class4_vectors)

centroid_class1 = np.mean(class1_vectors, axis=0)
centroid_class2 = np.mean(class2_vectors, axis=0)

spread_class1 = np.std(class1_vectors, axis=0)
spread_class2 = np.std(class2_vectors, axis=0)

distance_between_centroids = np.linalg.norm(centroid_class1 - centroid_class2)

print("Centroid of class 1:", centroid_class1)
print("Centroid of class 2:", centroid_class2)
print("Spread of class 1:", spread_class1)
print("Spread of class 2:", spread_class2)
print("Distance between centroids:", distance_between_centroids)

selected_feature = '0'  

feature_data = np.concatenate((class1_vectors, class2_vectors, class3_vectors, class4_vectors))

num_bins = 20  

hist, bins = np.histogram(feature_data, bins=num_bins)

plt.hist(feature_data, bins=num_bins, color='blue', alpha=0.7)


plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.title('Histogram of Feature: {}'.format(selected_feature))

plt.show()

mean_value = np.mean(feature_data)
variance_value = np.var(feature_data)

print("Mean of the feature:", mean_value)
print("Variance of the feature:", variance_value)



