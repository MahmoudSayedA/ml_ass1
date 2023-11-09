# first thing omar you need to install the upcoming lib's 
# pandas numpy scikit-learn matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# step-one Load the "loan_old.csv" dataset
loan_old_data = pd.read_csv("loan_old.csv")

# step-two Perform analysis on the dataset
# i) Check for missing values
missing_values = loan_old_data.isnull().sum()
print("Missing Values:\n", missing_values)

# ii) Check the type of each feature
print("Data Types:\n", loan_old_data.dtypes)

# iii) Check whether numerical features have the same scale (just print them to check)
numerical_columns = loan_old_data.select_dtypes(include=[np.number]).columns
print("Numerical Columns:\n", numerical_columns)

# iv) Visualize a pairplot between numerical columns
sns.pairplot(loan_old_data[numerical_columns])
plt.show()

# step-three Preprocess the data
# i) Remove records containing missing values
loan_old_data = loan_old_data.dropna()

# # test
# print(loan_old_data.info())

# ii) Separate features and targets
X = loan_old_data.drop(columns=["Loan_ID", "Max_Loan_Amount", "Loan_Status"])
y_loan_amount = loan_old_data["Max_Loan_Amount"]
y_loan_status = loan_old_data["Loan_Status"]

# iii) Shuffle and split the data into training and testing sets
X_train, X_test, y_train_loan_amount, y_test_loan_amount, y_train_loan_status, y_test_loan_status = train_test_split(
    X, y_loan_amount, y_loan_status, test_size=0.3, random_state=42)

# # test
# print(len(X_train), len(X_test))

# iv) encode categorical features
X_train_encoded = pd.get_dummies(X_train, columns=["Gender", "Married", "Dependents", "Education", "Credit_History", "Property_Area"], drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=["Gender", "Married", "Dependents", "Education", "Credit_History", "Property_Area"], drop_first=True)

# # test
# print(X_train_encoded.info())
# print(X_test_encoded.info())
# print(X_train_encoded)

# v) Encode categorical target
y_train_loan_status = y_train_loan_status.map({'Y': 1, 'N': 0})
y_test_loan_status = y_test_loan_status.map({'Y': 1, 'N': 0})

# vi) Numerical features standardization
numerical_columns = X_train_encoded.select_dtypes(include=[np.number]).columns
mean_values = X_train_encoded[numerical_columns].mean()
std_values = X_train_encoded[numerical_columns].std()
# Normalize
X_train_encoded[numerical_columns] = (X_train_encoded[numerical_columns] - mean_values) / std_values
X_test_encoded[numerical_columns] = (X_test_encoded[numerical_columns] - mean_values) / std_values


# step-four Fit a linear regression model to predict the loan amount
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train_encoded, y_train_loan_amount)

# step-five Evaluate the linear regression model using R-squared score
y_pred_loan_amount = linear_reg_model.predict(X_test_encoded)
r2 = r2_score(y_test_loan_amount, y_pred_loan_amount)
print("R^squared Score for Linear Regression:", r2)


# there are a problem with sigmoid method with the datatype of (x) .. take a look of it

# # f) Fit a logistic regression model from scratch using gradient descent
# def sigmoid(x):
#     try:
#         return 1 / (1 + np.exp(-x))
#     except AttributeError:
#         return 1 / (1 + np.exp(-np.array(x)))

# def logistic_regression(X, y, learning_rate, num_iterations):
#     m, n = X.shape
#     X = np.column_stack((np.ones((m, 1)), X))
#     theta = np.zeros(n + 1)
    
#     for _ in range(num_iterations):
#         z = np.dot(X, theta)
#         h = sigmoid(z)
#         gradient = np.dot(X.T, (h - y)) / m
#         theta -= learning_rate * gradient
    
#     return theta

# # Now i will test logistic_reg. method
# learning_rate = 0.01
# num_iterations = 1000
# theta_loan_status = logistic_regression(X_train_encoded, y_train_loan_status, learning_rate, num_iterations)

# # g) Function to calculate accuracy from scratch
# def calculate_accuracy(X, y, theta):
#     X = np.column_stack((np.ones((X.shape[0], 1)), X))
#     z = np.dot(X, theta)
#     h = sigmoid(z)
#     predicted_labels = int(h >= 0.5)
#     accuracy = np.mean(predicted_labels == y)
#     return accuracy


# # Calculate accuracy on the test set
# X_test_with_intercept = np.column_stack((np.ones((X_test_encoded.shape[0], 1)), X_test_encoded))
# accuracy_test = calculate_accuracy(X_test_encoded, y_test_loan_status, theta_loan_status)
# print("Test Accuracy for Logistic Regression:", accuracy_test)

