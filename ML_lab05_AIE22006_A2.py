import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_excel("C:/Users/jayan/Downloads/B TECH/Sem_4/ML/PurchasedData.xlsx", sheet_name="Purchase data")

# Mark customers with payments above Rs. 200 as RICH and others as POOR
data['Class'] = np.where(data['Payment (Rs)'] > 200, 'RICH', 'POOR')

# Convert 'Class' column to numerical values (1 for RICH, 0 for POOR)
data['Class'] = data['Class'].map({'RICH': 1, 'POOR': 0})

# Segregate data into matrices A & C
A = data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = data['Class'].values

# Splitting data into features (A) and target variable (C)
X_train, X_test, y_train, y_test = train_test_split(A, C, test_size=0.2, random_state=42)

# Initializing and training the classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Get predicted labels (RICH or POOR) for test data
predicted_labels = classifier.predict(X_test)

# Assuming that 'Cost of each product available for sale' is the predicted prices
cost_per_product = classifier.coef_[0]
predicted_prices = np.dot(X_test, cost_per_product)

# Convert X_test to a DataFrame
X_test_df = pd.DataFrame(X_test, columns=['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)'])

# Get the index of X_test_df
test_indices = X_test_df.index

# Use the indices to select 'Payment (Rs)' from the original data DataFrame
actual_prices = data.loc[test_indices, 'Payment (Rs)'].values

# Calculate evaluation metrics
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) score:", r2)
print("Mean Absolute Percentage Error (MAPE):", mape)
