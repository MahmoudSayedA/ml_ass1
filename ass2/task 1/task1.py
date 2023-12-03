import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

drag_data = pd.read_csv("ass2/task 1/drug.csv")

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

def calc_accuracy(prediction, actual):
    return np.mean(prediction == actual)

# First experiment
print(f'{"#" * 50} part 1 for fixed train-test splits')
# Determine the indices for fixed train-test splits
accuracies = []
tree_sizes = []

# # 1- We can split them to different chunks manually
# num_of_samples = len(X_encoded)
# split_indices1 = [
#     # Test from 70% to end and train is the rest
#     (
#         list(range(0, int(num_of_samples * 0.7))), 
#         list(range(int(num_of_samples * 0.7), num_of_samples))
#     ), 
#     # Test from 50% to 80% and train is the rest
#     (
#         list(range(0, int(num_of_samples * 0.5))) + list(range(int(num_of_samples * 0.8), num_of_samples)), 
#         list(range(int(num_of_samples * 0.5), int(num_of_samples * 0.8)))
#     ),
#     # Test from 30% to 60% and train is the rest
#     (
#         list(range(0, int(num_of_samples * 0.3))) + list(range(int(num_of_samples * 0.6), num_of_samples)), 
#         list(range(int(num_of_samples * 0.3), int(num_of_samples * 0.6)))
#     ),
#     # Test from 20% to 50% and train is the rest
#     (
#         list(range(0, int(num_of_samples * 0.2))) + list(range(int(num_of_samples * 0.5), num_of_samples)), 
#         list(range(int(num_of_samples * 0.2), int(num_of_samples * 0.5)))
#     ),
#     # Test is first 30% and train is the rest
#     (
#         list(range(int(num_of_samples * 0.3), num_of_samples)),
#         list(range(0, int(num_of_samples * 0.3)))
#     ),
# ]
# # Loop on indices to ge train and test indices
# for i, (train_indices, test_indices) in enumerate(split_indices1):
#     # Get tain and test data
#     X_encoded_train, X_encoded_test = X_encoded.iloc[train_indices], X_encoded.iloc[test_indices]
#     Y_encoded_train, Y_encoded_test = Y_encoded.iloc[train_indices], Y_encoded.iloc[test_indices]
    
#     # Fit the DecisionTree model
#     decision_tree_model = DecisionTreeClassifier()
#     decision_tree_model.fit(X_encoded_train, Y_encoded_train)
#     prediction = decision_tree_model.predict(X_encoded_test)

#     accuracy = calc_accuracy(prediction, Y_encoded_test)
#     accuracies.append(accuracy)

#     tree_sizes.append(decision_tree_model.tree_.node_count)

#     print(f"Experiment {i+1}: Tree Size - {decision_tree_model.tree_.node_count}, Accuracy - {accuracy * 100:.2f}%")

# 2- Or we can use train_test_split method if we want
iterations = 5
for i in range(iterations):
    # Get tain and test data
    X_encoded_train, X_encoded_test, Y_encoded_train, Y_encoded_test = train_test_split(X_encoded, Y_encoded, test_size=0.3, random_state=i*13)
    
    # Fit the DecisionTree model
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(X_encoded_train, Y_encoded_train)
    prediction = decision_tree_model.predict(X_encoded_test)

    accuracy = calc_accuracy(prediction, Y_encoded_test)
    accuracies.append(accuracy)

    tree_sizes.append(decision_tree_model.tree_.node_count)

    print(f"Experiment {i+1}: Tree Size - {decision_tree_model.tree_.node_count}, Accuracy - {accuracy * 100:.2f}%")

# Print the beset experiment
best_experiment = np.argmax(accuracies)
best_accuracy = accuracies[best_experiment]
best_tree_size = tree_sizes[best_experiment]

print(f"\nBest Model (Experiment {best_experiment+1}):")
print(f"Tree Size - {best_tree_size}, Accuracy - {best_accuracy * 100:.2f}%")


# Second experiment
print(f'{"#" * 50} part 2 for different training set sizes')
# Perform experiments for different training set sizes
train_sizes = list(range(30, 71, 10))  # Training set sizes from 30% to 70%
num_experiments = 5  # Number of experiments for each training set size

statistics = {
    'train_size': [],
    'mean_accuracy': [],
    'max_accuracy': [],
    'min_accuracy': [],
    'mean_tree_size': [],
    'max_tree_size': [],
    'min_tree_size': []
}

for size in train_sizes:
    accuracies = []
    tree_sizes = []
    
    for _ in range(num_experiments):
        # Split data into train and test sets with the current training set size
        X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y_encoded, test_size=(100 - size) / 100, random_state=_)
        
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(X_train, Y_train)
        prediction = decision_tree_model.predict(X_test)
        
        accuracy = np.mean(prediction == Y_test)
        accuracies.append(accuracy)
        
        tree_sizes.append(decision_tree_model.tree_.node_count)
    
    # Calculate statistics for the current training set size
    statistics['train_size'].append(size)
    statistics['mean_accuracy'].append(np.mean(accuracies))
    statistics['max_accuracy'].append(np.max(accuracies))
    statistics['min_accuracy'].append(np.min(accuracies))
    statistics['mean_tree_size'].append(np.mean(tree_sizes))
    statistics['max_tree_size'].append(np.max(tree_sizes))
    statistics['min_tree_size'].append(np.min(tree_sizes))

# Print statistics report
report = pd.DataFrame(statistics)
print(report)

# Create plots
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(statistics['train_size'], statistics['mean_accuracy'], label='Mean Accuracy')
plt.plot(statistics['train_size'], statistics['max_accuracy'], label='Max Accuracy')
plt.plot(statistics['train_size'], statistics['min_accuracy'], label='Min Accuracy')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Training Set Size')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(statistics['train_size'], statistics['mean_tree_size'], label='Mean Tree Size')
plt.plot(statistics['train_size'], statistics['max_tree_size'], label='Max Tree Size')
plt.plot(statistics['train_size'], statistics['min_tree_size'], label='Min Tree Size')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Tree Size')
plt.title('Tree Size vs Training Set Size')
plt.legend()

plt.tight_layout()
plt.show()






# Test the algorithm
# Divide data to train and test
# X_encoded_train, X_encoded_test, Y_encoded_train, Y_encoded_test = train_test_split(X_encoded, Y_encoded, test_size=0.3, random_state=1, shuffle=True)

# # Train the model and fit the data
# decision_tree_model = DecisionTreeClassifier()
# decision_tree_model.fit(X_encoded_train,Y_encoded_train)
# prediction = decision_tree_model.predict(X_encoded_test)


# # test the accuracy
# accuracy = calc_accuracy(prediction, Y_encoded_test)
# print(f"Test Accuracy: {accuracy * 100} %")