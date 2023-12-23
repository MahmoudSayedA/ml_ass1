import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset from CSV
mnist_train_data = pd.read_csv("mnist_train.csv")

# Number of unique classes
num_classes = mnist_train_data['label'].nunique()
print("Number of unique classes:", num_classes)

# Number of features
num_features = len(mnist_train_data.columns) - 1  # Subtract 1 for the label column
print("Number of features:", num_features)

# Check for missing values
missing_values = mnist_train_data.isnull().sum().sum()
print("Number of missing values:", missing_values)

# Separate labels and pixel values
labels = mnist_train_data['label']
pixels = mnist_train_data.drop('label', axis=1)

# Normalize each image by dividing each pixel by 255
# Every value now is between 0 and 1 and this is a common normalization for images
normalized_pixels = pixels / 255.0

# Resize images to dimensions of 28 by 28
# a collection of matrices every matrix represents an image 28x28
# convert it to a 1D array to split and use it in the ML model
resized_pixels = np.array([np.array(pixel).reshape(28, 28).flatten() for pixel in normalized_pixels.values])

# Visualize some resized images
num_samples = 5
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(resized_pixels[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')
plt.show()

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    resized_pixels, labels, test_size=0.2, random_state=562)

# Create a K-NN classifier
knn = KNeighborsClassifier()

# Define the hyperparameter grid to search
param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}

#! grid search takes amout of time to end..
# Perform grid search using cross-validation
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the K-NN classifier with the best hyperparameters on the training set
best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])
#! this is best params from grid search
# best_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
best_knn.fit(x_train, y_train)

# Predictions on the validation set
y_val_pred = best_knn.predict(x_val)

# Evaluate the performance on the validation set
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy*100:.3f}%")
