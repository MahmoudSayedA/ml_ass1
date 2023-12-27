# second architecture
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

# Load data
train_data = pd.read_csv('/content/mnist_train.csv')
test_data = pd.read_csv('/content/mnist_test.csv')

# Drop unnecessary columns
X = train_data.drop(columns=['label'])
y = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Data Preprocessing
# Number of unique classes
num_classes = y.nunique()
print("Number of unique classes:", num_classes)

# Number of features
num_features = len(X.columns)
print("Number of features:", num_features)

# Check for missing values
missing_values = train_data.isnull().sum().sum()
print("Number of missing values:", missing_values)

# Normalize pixel values
X_normalized = X / 255.0
X_test_normalized = X_test / 255.0

# Resize images to dimensions of 28 by 28
# Flatten feature data
X_flattened = X_normalized.values.reshape(-1, 28, 28)
X_test_flattened = X_test_normalized.values.reshape(-1, 28, 28)

# Visualize some resized images
plt.figure(figsize=(10, 4))
for i in range(1, 6):
    plt.subplot(1, 5, i)
    plt.imshow(X_flattened[i], cmap='gray')  # Corrected line
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# One-hot encode labels before splitting
y_train_encoded = to_categorical(y_train, num_classes=10)
y_val_encoded = to_categorical(y_val, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# Define 1st architecture ANN model
model_ann1st = models.Sequential()
model_ann1st.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
model_ann1st.add(layers.Dense(64, activation='relu'))
model_ann1st.add(layers.Dense(10, activation='softmax'))

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# Compile the ANN model
model_ann1st.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the ANN model
model_ann1st.fit(X_train, y_train_encoded, epochs=5, batch_size=64, validation_data=(X_val, y_val_encoded))


# Evaluate The ANN model on the validation set
y_val_pred_probs1st = model_ann1st.predict(X_val)
y_val_pred = np.argmax(y_val_pred_probs1st, axis=1)

# Confusion Matrix
conf_mat = confusion_matrix(y_val, y_val_pred)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate The 2nd architecture ANN model on the validation set
val_loss1st, val_acc1st = model_ann1st.evaluate(X_val, y_val_encoded)
print("Validation accuracy:", val_acc1st)

# Evaluate the ANN model on the test set
# test_loss, test_acc = model_ann1st.evaluate(X_test_normalized, y_test_encoded)
# print("Test accuracy:", test_acc)

# Define an ANN model with more hidden layers
model_ann2nd = models.Sequential()
model_ann2nd.add(layers.Dense(64, activation='relu', input_shape=(28 * 28,)))
model_ann2nd.add(layers.Dense(32, activation='relu'))
# model_ann2nd.add(layers.Dense(32, activation='relu'))  # Additional layer
# model_ann2nd.add(layers.Dense(16, activation='relu'))  # Additional layer
model_ann2nd.add(layers.Dense(10, activation='softmax'))

# Decrease the learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Compile the ANN model with the custom optimizer
model_ann2nd.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Increase the batch size and train the ANN model
model_ann2nd.fit(X_train, y_train_encoded, epochs=5, batch_size=128, validation_data=(X_val, y_val_encoded))

# Evaluate The ANN model on the validation set
y_val_pred_probs2nd = model_ann2nd.predict(X_val)
y_val_pred = np.argmax(y_val_pred_probs2nd, axis=1)

# Create a K-NN classifier
# knn = KNeighborsClassifier()

# # Define the hyperparameter grid to search
# param_grid = {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}

# #! grid search takes amout of time to end..
# # Perform grid search using cross-validation
# grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters from the grid search
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
best_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
best_knn.fit(X_train, y_train)
# Predictions on the validation set
y_preddiction = best_knn.predict(X_val)

# Evaluate the performance on the validation set
accuracyKnn = accuracy_score(y_val, y_preddiction)

# Confusion Matrix
conf_mat = confusion_matrix(y_val, y_val_pred)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate The ANN model on the validation set
val_loss, val_acc2nd = model_ann2nd.evaluate(X_val, y_val_encoded)
print("Validation accuracy:", val_acc2nd)

print ("The 1st architecture ANN model accuracy is", val_acc1st)
print ("The 2nd architecture ANN model accuracy is", val_acc2nd)
best_accuracy = accuracyKnn
best_model = None
print("knn accuracy: ", accuracyKnn)
print("ANN first: ", val_acc1st)
print("ANN second: ", val_acc2nd)
if (val_acc2nd > val_acc1st):
    best_model = "2nd architecture of ANN"
    best_accuracy = val_acc2nd
    test_loss, test_acc = model_ann2nd.evaluate(X_test_normalized, y_test_encoded)
    print("The 2nd architecture ANN model is better than the 1st architecture ANN model.")
else: 
    best_model = "1st architecture of ANN"
    best_accuracy = val_acc1st
    test_loss, test_acc = model_ann1st.evaluate(X_test_normalized, y_test_encoded)
    print("The 1st architecture ANN model is better than the 2nd architecture ANN model.")
if (best_accuracy * 100 >= accuracyKnn * 100):
    print("The best model is ANN")
elif (best_accuracy * 100 < accuracyKnn * 100):
    best_model = "KNN"
    best_accuracy = accuracyKnn
    best_knn.fit(X_train, y_train)
    y_pred = best_knn.predict(X_test_normalized)
    test_acc = accuracy_score(y_test, y_pred)
    print("The best model is KNN")
print (best_accuracy)
print ("best model: " + best_model)
# Evaluate the ANN model on the test set
print("Test accuracy:", test_acc)