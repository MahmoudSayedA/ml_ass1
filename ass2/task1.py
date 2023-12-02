import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

drag_data = pd.read_csv("ass2/drug.csv")

# Step-1 data preprocessing
# Remove records with missing values
missing_values = drag_data.isnull().sum()
print("Missing_values:\n", missing_values)
drag_data = drag_data.dropna()

# Split data into X,Y
X = drag_data.drop(columns=["Drug"])
Y = drag_data["Drug"]

# Encoding categorical values
X_encoded = pd.get_dummies(X, columns=['Sex', 'BP', 'Cholesterol',])
Y_encoded = Y.map({
    "drugA":1,
    "drugB":2,
    "drugC":3,
    "drugX":4,
    "drugY":5
})

# test
# print(X_encoded, Y_encoded)

# Normalize numerical data
numerical_columns = drag_data.select_dtypes(include=[np.number]).columns
mean_values = X_encoded[numerical_columns].mean()
std_values = X_encoded[numerical_columns].std()
# Normalize
X_encoded[numerical_columns] = (X_encoded[numerical_columns] - mean_values) / std_values

print(X_encoded[1])



# First experiment
# 
# Second experiment
# 

# Divide data to train and test
X_encoded_train, X_encoded_test, Y_encoded_train, Y_encoded_test = train_test_split(X_encoded, Y_encoded, test_size=0.3, random_state=1, shuffle=True)

# Train the model and fit the data
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_encoded_train,Y_encoded_train)
prediction = decision_tree_model.predict(X_encoded_test)

def calc_accuracy(prediction, actual):
    return np.mean(prediction == actual)

# test the accuracy
accuracy = calc_accuracy(prediction, Y_encoded_test)
print(f"Test Accuracy: {accuracy * 100} %")